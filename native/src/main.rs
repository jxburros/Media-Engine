//! media-engine-native: the owned native engine (`backend_id: native-sdxl`, `ownership:
//! engine`) implementing media-worker/1.0–1.3 for the sdxl-base family.
//!
//! The launch contract is the Python worker's: `--probe`, `--version`, one JSON startup record
//! on stdout, `--catalog`, a private `--directory`, the credential in `MEDIA_ENGINE_TOKEN`,
//! `--port 0`, `--device cpu|cuda:N`, SIGTERM/SIGINT and the supervised-stdin watch.

mod contracts;
mod http;
mod json;
mod sha256;
mod sys;
mod worker;

use contracts::{fail, WorkerError};
use json::Value;
use std::collections::BTreeMap;
use std::io::Read;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use worker::{error_body, Worker};

const HELP: &str = "usage: media-engine-native [--version] [--probe] [--port PORT] [--directory DIR] --catalog FILE
                           [--device cpu|cuda:N] [--backend native-sdxl] [--memory-budget-mb MB]
                           [--max-retained N] [--max-storage-mb MB] [--max-input-mb MB]
                           [--bind 127.0.0.1|::1] [--no-parent-watch]

Owned native media-worker engine (native-sdxl). The credential is read from MEDIA_ENGINE_TOKEN
(at least 32 characters). Network mode (--tls-*) is not implemented by this engine yet: it binds
loopback only.";

fn usage_error(message: &str) -> ! {
    eprintln!("{}\nmedia-engine-native: error: {}", HELP.lines().next().unwrap(), message);
    std::process::exit(2);
}

struct Args {
    probe: bool,
    port: u16,
    directory: PathBuf,
    catalog: Option<PathBuf>,
    device: String,
    memory_budget_mb: Option<u64>,
    max_retained: usize,
    max_storage_mb: u64,
    max_input_mb: u64,
    bind: String,
    no_parent_watch: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        probe: false,
        port: 7811,
        directory: PathBuf::from(".engine"),
        catalog: None,
        device: "cuda:0".into(),
        memory_budget_mb: None,
        max_retained: 128,
        max_storage_mb: 1024,
        max_input_mb: 32,
        bind: "127.0.0.1".into(),
        no_parent_watch: false,
    };
    let mut it = std::env::args().skip(1);
    fn number<T: std::str::FromStr>(flag: &str, value: Option<String>) -> T {
        value.and_then(|v| v.parse().ok()).unwrap_or_else(|| usage_error(&format!("argument {}: expected a number", flag)))
    }
    while let Some(arg) = it.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = || inline.clone().or_else(|| it.next());
        match flag.as_str() {
            "--help" | "-h" => {
                println!("{}", HELP);
                std::process::exit(0);
            }
            "--version" => {
                println!("media-engine-native {}", contracts::WORKER_VERSION);
                std::process::exit(0);
            }
            "--probe" => args.probe = true,
            "--port" => args.port = number("--port", value()),
            "--directory" => args.directory = PathBuf::from(value().unwrap_or_else(|| usage_error("argument --directory: expected one argument"))),
            "--catalog" => args.catalog = Some(PathBuf::from(value().unwrap_or_else(|| usage_error("argument --catalog: expected one argument")))),
            "--device" => args.device = value().unwrap_or_else(|| usage_error("argument --device: expected one argument")),
            "--backend" => {
                let v = value().unwrap_or_default();
                if v != worker::BACKEND_ID {
                    usage_error(&format!("argument --backend: invalid choice: '{}' (choose from 'native-sdxl')", v));
                }
            }
            "--memory-budget-mb" => args.memory_budget_mb = Some(number("--memory-budget-mb", value())),
            "--max-retained" => args.max_retained = number("--max-retained", value()),
            "--max-storage-mb" => args.max_storage_mb = number("--max-storage-mb", value()),
            "--max-input-mb" => args.max_input_mb = number("--max-input-mb", value()),
            "--bind" => args.bind = value().unwrap_or_else(|| usage_error("argument --bind: expected one argument")),
            "--no-parent-watch" => args.no_parent_watch = true,
            "--tls-cert" | "--tls-key" | "--tls-client-ca" => {
                usage_error("network mode (mutual TLS) is not implemented by the native engine; bind loopback")
            }
            other => usage_error(&format!("unrecognized arguments: {}", other)),
        }
    }
    args
}

fn respond(response: &mut http::Response, status: u16, value: &Value) {
    let _ = response.send(status, "application/json", json::to_string(value).as_bytes());
}

fn respond_error(response: &mut http::Response, e: &WorkerError) {
    respond(response, e.status, &obj! {"error" => error_body(e.code, e.field.as_deref())});
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for (i, x) in a.iter().enumerate() {
        diff |= (*x ^ b.get(i % b.len().max(1)).copied().unwrap_or(0)) as usize;
    }
    diff == 0
}

fn body_json(request: &mut http::Request) -> Result<Value, WorkerError> {
    let data = request.body(4 * 1024 * 1024).map_err(|status| fail(status, "invalid_request", None))?;
    json::parse(&data).map_err(|_| fail(400, "invalid_request", None))
}

fn stream_events(worker: &Worker, request: &http::Request, response: &mut http::Response, execution_id: &str) -> Result<(), WorkerError> {
    let (_, last, _) = worker.events_after(execution_id, 0)?;
    let cursor: i64 = request.header("last-event-id").unwrap_or("0").trim().parse().map_err(|_| fail(400, "invalid_request", None))?;
    if cursor < 0 || cursor as usize > last {
        return Err(fail(409, "resume_required", None));
    }
    let mut cursor = cursor as usize;
    if response.head(200, &[("Content-Type", "text/event-stream".into()), ("Cache-Control", "no-store".into())]).is_err() {
        return Ok(());
    }
    loop {
        let (events, last, terminal) = worker.events_after(execution_id, cursor)?;
        let mut chunk = String::new();
        for event in &events {
            chunk.push_str(&format!(
                "id: {}\nevent: {}\ndata: {}\n\n",
                event["sequence"].as_int().unwrap_or(0),
                event["type"].as_str().unwrap_or(""),
                json::to_string(event)
            ));
        }
        cursor = last;
        if !chunk.is_empty() && response.write(chunk.as_bytes()).is_err() {
            return Ok(());
        }
        if terminal {
            return Ok(());
        }
        if response.write(b": heartbeat\n\n").is_err() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

fn route(worker: &Arc<Worker>, token: &str, mut request: http::Request, response: &mut http::Response) -> Result<(), WorkerError> {
    let method = request.method.clone();
    let path = request.path.clone();
    if request.header("origin").is_some() {
        return Err(fail(403, "browser_origin_denied", None));
    }
    if path == "/health" && method == "GET" {
        let error = worker.is_error();
        respond(response, if error { 503 } else { 200 }, &obj! {"status" => if error { "error" } else { "ok" }});
        return Ok(());
    }
    let expected = format!("Bearer {}", token);
    if !constant_time_eq(request.header("authorization").unwrap_or("").as_bytes(), expected.as_bytes()) {
        return Err(fail(401, "unauthorized", None));
    }
    match (method.as_str(), path.as_str()) {
        ("GET", "/v1/info") => return Ok(respond(response, 200, &worker.info())),
        ("GET", "/v1/capabilities") => return Ok(respond(response, 200, &worker.capabilities())),
        ("POST", "/v1/validate") => {
            let body = body_json(&mut request)?;
            let value = if body.get("bundle").is_some() {
                worker.validation(&body)?
            } else if body.get("adapters").is_some() {
                worker.validate_adapters_only(&body)?
            } else {
                worker.validate_execution(&body)?
            };
            return Ok(respond(response, 200, &value));
        }
        ("POST", "/v1/model/load") | ("POST", "/v1/model/unload") => {
            let body = body_json(&mut request)?;
            let action = path.rsplit('/').next().unwrap();
            return Ok(respond(response, 202, &worker.control(action, body)?));
        }
        ("POST", "/v1/inputs") => {
            if request.header("content-type") != Some("image/png") {
                return Err(contracts::bad("unsupported", "image"));
            }
            let limit = (worker.max_input_mb * 1024 * 1024) as usize;
            let data = request.body(limit).map_err(|status| fail(status, "invalid_request", None))?;
            return Ok(respond(response, 201, &worker.store_input(&data)?));
        }
        ("POST", "/v1/executions") => {
            let body = body_json(&mut request)?;
            return Ok(respond(response, 202, &worker.submit(body)?));
        }
        ("GET", "/v1/executions") => {
            let attempt = http::query_value(&request.query, "attempt_id").unwrap_or_default();
            return Ok(respond(response, 200, &worker.attempt(&attempt)?));
        }
        _ => {}
    }
    if let Some(operation_id) = path.strip_prefix("/v1/operations/") {
        let mut s = worker.lock();
        let op = s.operations.get(operation_id).cloned().ok_or_else(|| fail(404, "operation_not_found", None))?;
        let running = op["state"].as_str() == Some("running");
        if method == "DELETE" && !running {
            s.operations.remove(operation_id);
            drop(s);
            return Ok(respond(response, 200, &obj! {"released" => true}));
        }
        drop(s);
        if method == "GET" {
            return Ok(respond(response, 200, &op));
        }
        return Err(fail(409, "busy", None));
    }
    if let Some(input_id) = path.strip_prefix("/v1/inputs/").filter(|i| contracts::is_input_id(i)) {
        return match method.as_str() {
            "GET" => Ok(respond(response, 200, &worker.input_record(input_id)?)),
            "DELETE" => {
                worker.release_input(input_id)?;
                Ok(respond(response, 200, &obj! {"released" => true}))
            }
            _ => Err(fail(404, "not_found", None)),
        };
    }
    let Some(rest) = path.strip_prefix("/v1/executions/") else { return Err(fail(404, "not_found", None)) };
    let (execution_id, suffix) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    let output_id = suffix.strip_prefix("/outputs/");
    let valid_output = output_id.is_some_and(|o| {
        (1..=32).contains(&o.len())
            && o.as_bytes()[0].is_ascii_lowercase()
            && o.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    });
    if !contracts::is_handle("exec_", execution_id) || !(suffix.is_empty() || suffix == "/cancel" || suffix == "/events" || valid_output) {
        return Err(fail(404, "not_found", None));
    }
    match (method.as_str(), suffix) {
        ("POST", "/cancel") => {
            let snap = worker.cancel(execution_id)?;
            let terminal = matches!(snap["state"].as_str(), Some("succeeded" | "failed" | "cancelled"));
            respond(response, if terminal { 200 } else { 202 }, &snap);
            Ok(())
        }
        ("DELETE", "") => {
            worker.release(execution_id)?;
            respond(response, 200, &obj! {"released" => true});
            Ok(())
        }
        ("GET", "") => {
            respond(response, 200, &worker.snapshot(execution_id)?);
            Ok(())
        }
        ("GET", "/events") => stream_events(worker, &request, response, execution_id),
        ("GET", _) if valid_output => {
            let (file, mime) = worker.output_path(execution_id, output_id.unwrap())?;
            let data = std::fs::read(&file).map_err(|_| fail(410, "output_unavailable", None))?;
            let _ = response.head(200, &[("Content-Type", mime), ("Content-Length", data.len().to_string())]);
            let _ = response.write(&data);
            Ok(())
        }
        _ => Err(fail(404, "not_found", None)),
    }
}

extern "C" {
    fn umask(mask: u32) -> u32;
}

fn main() {
    unsafe {
        umask(0o077);
    }
    let args = parse_args();
    if args.probe {
        println!(
            "{}",
            json::to_string(&obj! {
                "protocol" => contracts::PROTOCOL,
                "supported_protocols" => json::strings(&contracts::SUPPORTED_PROTOCOLS),
                "worker_version" => contracts::WORKER_VERSION,
                "backend_id" => worker::BACKEND_ID,
                "backends" => json::strings(&[worker::BACKEND_ID]),
                "commands" => Value::Array(vec![]),
            })
        );
        return;
    }
    let bind: std::net::IpAddr = args.bind.parse().unwrap_or_else(|_| usage_error("--bind must be an IP address literal"));
    if !bind.is_loopback() {
        usage_error("a non-loopback address is only served over mutual TLS, which the native engine does not implement yet");
    }
    let device_ok = args.device == "cpu" || args.device.strip_prefix("cuda:").is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
    if !device_ok {
        usage_error("device must be cpu or cuda:N");
    }
    if args.memory_budget_mb == Some(0) || args.max_retained == 0 || args.max_storage_mb == 0 || args.max_input_mb == 0 {
        usage_error("resource limits must be positive");
    }
    let Some(catalog_path) = &args.catalog else { usage_error("--catalog is required") };
    let catalog: BTreeMap<String, String> = std::fs::read(catalog_path)
        .ok()
        .and_then(|d| json::parse(&d).ok())
        .and_then(|v| {
            v.as_object().and_then(|m| {
                m.iter()
                    .map(|(k, v)| v.as_str().filter(|p| p.starts_with('/')).map(|p| (k.clone(), p.to_string())))
                    .collect::<Option<BTreeMap<_, _>>>()
            })
        })
        .unwrap_or_else(|| usage_error("catalog must map opaque handles to absolute local directories"));
    let token = std::env::var("MEDIA_ENGINE_TOKEN").unwrap_or_default();
    if token.len() < 32 {
        usage_error("Set MEDIA_ENGINE_TOKEN to at least 32 random characters");
    }
    let transport = obj! {
        "scheme" => "http",
        "bind" => bind.to_string(),
        "certificate_sha256" => Value::Null,
        "ownership_available" => !args.no_parent_watch,
    };
    let backend = worker::Backend::new(&args.device, args.memory_budget_mb);
    // A private per-session directory prevents another session's outputs being collected.
    let directory = args.directory.join(sys::random_hex());
    let worker = Arc::new(
        Worker::new(directory, backend, catalog, args.max_retained, args.max_storage_mb, args.max_input_mb, transport)
            .unwrap_or_else(|e| usage_error(&format!("cannot create the work directory: {}", e))),
    );
    let listener = TcpListener::bind((bind, args.port)).unwrap_or_else(|e| {
        eprintln!("media-engine-native: cannot bind: {}", e);
        std::process::exit(1);
    });
    let port = listener.local_addr().map(|a| a.port()).unwrap_or(args.port);

    sys::install_stop_handlers();
    if std::env::var("MEDIA_ENGINE_SUPERVISED").as_deref() == Ok("1") && !args.no_parent_watch {
        // The supervisor owns stdin; EOF means its process exited (including a crash).
        std::thread::spawn(|| {
            let mut sink = Vec::new();
            let _ = std::io::stdin().read_to_end(&mut sink);
            sys::STOP.store(true, Ordering::SeqCst);
        });
    }
    {
        let worker = worker.clone();
        std::thread::spawn(move || loop {
            if sys::STOP.load(Ordering::SeqCst) {
                worker.stop();
                let deadline = std::time::Instant::now() + Duration::from_secs(5);
                while worker.is_active() && std::time::Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(100));
                }
                std::process::exit(0);
            }
            std::thread::sleep(Duration::from_millis(100));
        });
    }

    let mut record = obj! {
        // The launch-record minor stays 1.1 for 1.1 launchers; callers negotiate from the list.
        "protocol" => "media-worker/1.1",
        "supported_protocols" => json::strings(&worker.protocols),
        "highest_protocol" => worker.protocols.last().unwrap().as_str(),
        "worker_version" => contracts::WORKER_VERSION,
        "backend_id" => worker::BACKEND_ID,
        "session_id" => worker.session_id.as_str(),
        "port" => port as u64,
        "pid" => std::process::id() as u64,
        "device_identity" => worker.device_identity(),
    };
    if let (Value::Object(m), Value::Object(t)) = (&mut record, &worker.transport) {
        m.extend(t.clone());
    }
    println!("{}", json::to_string(&record));

    let token = Arc::new(token);
    http::serve(listener, move |request, mut response| {
        if let Err(e) = route(&worker, &token, request, &mut response) {
            respond_error(&mut response, &e);
        }
    });
}
