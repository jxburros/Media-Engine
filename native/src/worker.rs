//! The media-worker state machine (`Worker` in `src/media_engine/server.py`) and the native
//! backend: one warm pipeline, single-flight executions, retained records, SSE events.

use crate::contracts::{self as c, bad, fail, strict, Result, WorkerError};
use crate::json::{strings, Map, Value};
use crate::obj;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

pub const BACKEND_ID: &str = "native-sdxl";
pub const OWNERSHIP: &str = "engine";
const TERMINAL: [&str; 3] = ["succeeded", "failed", "cancelled"];

pub fn error_body(code: &str, field: Option<&str>) -> Value {
    let mut e = obj! {"code" => code, "message" => code.replace('_', " "), "retryable" => false};
    if let (Some(f), Value::Object(m)) = (field, &mut e) {
        m.insert("field".into(), Value::str(f));
    }
    if code == "out_of_memory" {
        if let Value::Object(m) = &mut e {
            m.insert("restart_required".into(), Value::Bool(true));
        }
    }
    e
}

fn is_cuda(device: &str) -> bool {
    device.starts_with("cuda:")
}

/// What the backend reports about itself and the loaded pipeline.
pub struct Backend {
    pub device: String,
    pub run_device: String,
    pub memory_budget_mb: Option<u64>,
    pub loaded_estimate: Option<Value>,
    pub applied_fallbacks: Vec<Value>,
    pub pipeline: Option<Arc<sdxl::Pipeline>>,
}

impl Backend {
    pub fn new(device: &str, memory_budget_mb: Option<u64>) -> Backend {
        Backend {
            device: device.to_string(),
            run_device: device.to_string(),
            memory_budget_mb,
            loaded_estimate: None,
            applied_fallbacks: Vec::new(),
            pipeline: None,
        }
    }

    /// CUDA execution exists only in a build with the `cuda` feature.
    pub fn device_available(&self) -> bool {
        self.device == "cpu" || (cfg!(feature = "cuda") && is_cuda(&self.device))
    }

    pub fn precisions(&self) -> Vec<&'static str> {
        if self.device == "cpu" { vec!["float32"] } else { vec!["float16", "float32"] }
    }

    pub fn device_identity(&self) -> Value {
        let (host_id, source) = crate::sys::host_identity();
        let cpu = self.device == "cpu";
        obj! {
            "device" => self.device.as_str(),
            "index" => if cpu { Value::Null } else { Value::Int(self.device[5..].parse().unwrap_or(0)) },
            "name" => if cpu { Value::str("cpu") } else { Value::Null },
            "total_memory_mb" => if cpu { Value::from(crate::sys::total_memory_mb()) } else { Value::Null },
            "host_id" => host_id,
            "host_id_source" => source,
            "pid" => std::process::id() as u64,
        }
    }

    pub fn capabilities(&self) -> Value {
        let available = self.device_available();
        let tasks = strings(&c::TASKS);
        obj! {
            "protocol" => c::PROTOCOL,
            "supported_protocols" => strings(&c::SUPPORTED_PROTOCOLS),
            "worker_version" => c::WORKER_VERSION,
            "backend_id" => BACKEND_ID,
            "ownership" => OWNERSHIP,
            "ready" => available || self.run_device == "cpu",
            "missing_dependencies" => Value::Array(vec![]),
            "missing_task_dependencies" => obj!(),
            "device" => self.device.as_str(),
            "device_available" => available,
            "memory_budget_mb" => self.memory_budget_mb,
            "memory_estimate" => self.loaded_estimate.clone().unwrap_or_else(|| obj! {
                "value_mb" => Value::Null,
                "source" => "unavailable",
                "uncertainty" => "no bundle validated or loaded in this session",
            }),
            // Compiled in, not discovered: the native binary's whole third-party set.
            "dependencies" => obj! {"candle-core" => "0.9.2", "png" => "0.18.1"},
            "max_concurrency" => 1i64,
            "optimizations" => Value::Array(vec![]),
            // bfloat16 runs on candle; nothing else is implemented by this engine yet.
            "unadvertised_optimizations" => Value::Array(vec![obj! {"option" => "precision", "value" => "bfloat16", "reason" => "no_benchmark"}]),
            "model_families" => strings(&[c::FAMILY]),
            "loaders" => Value::Array(vec![obj! {
                "format" => "diffusers-safetensors",
                "families" => strings(&[c::FAMILY]),
                "roles" => strings(&c::ROLES),
                "tasks" => tasks.clone(),
                "ownership" => OWNERSHIP,
            }]),
            "tasks" => tasks,
            "task_parameters" => obj! {
                "text-to-image" => obj! {
                    "inputs" => strings(&["prompt", "negative_prompt"]),
                    "parameters" => strings(&["count"]),
                    "defaults" => obj! {"count" => 1i64},
                    "outputs" => strings(&["image"]),
                    "output_formats" => obj! {"image" => "image/png"},
                    "repeated_outputs" => strings(&["image"]),
                    "families" => strings(&[c::FAMILY]),
                },
                "image-to-image" => obj! {
                    "inputs" => strings(&["prompt", "negative_prompt", "image"]),
                    "parameters" => strings(&["strength"]),
                    "input_formats" => strings(&["image/png"]),
                    "outputs" => strings(&["image"]),
                    "output_formats" => obj! {"image" => "image/png"},
                    "families" => strings(&[c::FAMILY]),
                },
            },
            "adapters" => obj! {
                "formats" => strings(&[c::ADAPTER_FORMAT]),
                "targets" => strings(&c::ADAPTER_TARGETS),
                "max_adapters" => c::MAX_ADAPTERS,
                "scale" => vec![Value::Int(0), Value::Int(2)],
                "identity" => "load",
                "families" => strings(&[c::FAMILY]),
            },
            "defaults" => c::defaults(),
            "task_defaults" => c::task_defaults(),
            "load_defaults" => obj! {
                "precision" => if self.device == "cpu" { "float32" } else { "float16" },
                "attention" => "sdpa",
                "offload" => "none",
            },
            "limits" => c::limits(),
            "precisions" => strings(&self.precisions()),
            "auto_options" => strings(&c::AUTO_OPTIONS),
            "fallbacks" => strings(&c::FALLBACKS),
            "formats" => strings(&c::FORMATS),
        }
    }

    pub fn compute_estimate(&self, root: &Path, precision: &str, adapters: &[c::ResolvedAdapter], device: &str) -> Result<Value> {
        let mut estimate = c::memory_estimate(root, precision, device)?;
        let per = c::precision_bytes(precision);
        let mut extra = 0u64;
        let list: Vec<Value> = adapters
            .iter()
            .map(|a| {
                let parameters = a.layout.get("parameters").and_then(Value::as_int).unwrap_or(0) as u64;
                let loaded = parameters * per / (1024 * 1024);
                extra += loaded;
                obj! {"adapter_id" => a.record.get("adapter_id").cloned().unwrap_or(Value::Null), "parameters" => parameters, "loaded_mb" => loaded}
            })
            .collect();
        if let Value::Object(m) = &mut estimate {
            let value = m["value_mb"].as_int().unwrap_or(0) + extra as i64;
            m.insert("value_mb".into(), Value::Int(value));
            m.insert("adapters".into(), Value::Array(list));
        }
        Ok(estimate)
    }

    fn budget_mb(&self) -> Option<u64> {
        // Device memory is not measurable without the CUDA build; the budget is the operator's.
        self.memory_budget_mb
    }

    /// `plan_load()`: resolve `auto` and authorized fallbacks before loading; no side effects.
    pub fn plan(&self, root: &Path, options: &Map, adapters: &[c::ResolvedAdapter], authorized: &[String]) -> Result<Plan> {
        let mut resolved = options.clone();
        if options["precision"].as_str() != Some("auto") && authorized.is_empty() {
            return Ok(Plan { options: resolved, resolutions: vec![], applied: vec![], device: self.device.clone() });
        }
        let available = self.device_available();
        let budget_mb = if available { self.budget_mb() } else { None };
        let estimate = |p: &str| -> Result<i64> {
            Ok(self.compute_estimate(root, p, adapters, &self.device)?["value_mb"].as_int().unwrap_or(0))
        };
        let precisions = self.precisions();
        let accelerator = self.device != "cpu";
        let budget = budget_mb.map(|b| format!("{} MiB budget", b));
        let mut resolutions = Vec::new();
        let mut applied = Vec::new();
        if options["precision"].as_str() == Some("auto") {
            let (value, reason) = if accelerator && precisions.contains(&"float16") {
                let e = estimate("float16")?;
                let reason = if !available {
                    "the selected device is unavailable; float16 is its lowest-memory advertised precision".to_string()
                } else if budget.is_none() {
                    format!("float16 is the lowest-memory precision this device advertises; estimate {} MiB, no budget or device memory measurable", e)
                } else if e as u64 <= budget_mb.unwrap() {
                    format!("float16 estimate {} MiB fits the {}", e, budget.clone().unwrap())
                } else {
                    format!("float16 estimate {} MiB exceeds the {}; it is the lowest-memory advertised precision", e, budget.clone().unwrap())
                };
                ("float16", reason)
            } else {
                ("float32", "the selected device advertises float32 only".to_string())
            };
            resolved.insert("precision".into(), Value::str(value));
            resolutions.push(obj! {"option" => "precision", "requested" => "auto", "resolved" => value, "reason" => reason});
        }
        let mut run_device = self.device.clone();
        if accelerator {
            let precision = resolved["precision"].as_str().unwrap().to_string();
            let mut over = available && budget_mb.is_some_and(|b| estimate(&precision).is_ok_and(|e| e as u64 > b));
            if over && authorized.iter().any(|a| a == "lower_precision") {
                let budget_value = budget_mb.unwrap();
                let lower: Vec<&str> = precisions
                    .iter()
                    .copied()
                    .filter(|p| c::precision_bytes(p) < c::precision_bytes(&precision) && estimate(p).is_ok_and(|e| e as u64 <= budget_value))
                    .collect();
                if let Some(first) = lower.first() {
                    applied.push(obj! {
                        "fallback" => "lower_precision",
                        "reason" => format!("{} estimate {} MiB exceeds the {}; {} needs {} MiB", precision, estimate(&precision)?, budget.clone().unwrap(), first, estimate(first)?),
                        "device" => self.device.as_str(),
                        "precision" => *first,
                    });
                    resolved.insert("precision".into(), Value::str(*first));
                    over = false;
                }
            }
            if (!available || over) && authorized.iter().any(|a| a == "cpu") {
                let reason = if !available {
                    "the selected device is unavailable".to_string()
                } else {
                    let p = resolved["precision"].as_str().unwrap().to_string();
                    format!("{} estimate {} MiB exceeds the {}", p, estimate(&p)?, budget.clone().unwrap())
                };
                resolved.insert("precision".into(), Value::str("float32"));
                run_device = "cpu".into();
                applied.push(obj! {"fallback" => "cpu", "reason" => reason, "device" => "cpu", "precision" => "float32"});
            }
        }
        Ok(Plan { options: resolved, resolutions, applied, device: run_device })
    }
}

pub struct Plan {
    pub options: Map,
    pub resolutions: Vec<Value>,
    pub applied: Vec<Value>,
    pub device: String,
}

struct Execution {
    attempt_id: String,
    execution_id: String,
    protocol: String,
    task: String,
    state: String,
    stage: String,
    progress: Value,
    events: Vec<Value>,
    effective_settings: Value,
    bundle_digest: String,
    implementation: Value,
    cancel_requested: bool,
    outputs: Vec<Value>,
    error: Value,
    applied_fallbacks: Option<Value>,
    measurements: Option<Value>,
    request: Value,
    cancel: Arc<AtomicBool>,
    image_path: Option<PathBuf>,
    expected_outputs: Vec<String>,
}

pub struct State {
    pub backend: Backend,
    pub state: String,
    pub loaded: Option<Value>,
    executions: BTreeMap<String, Execution>,
    attempts: BTreeMap<String, String>,
    pub operations: BTreeMap<String, Value>,
    inputs: BTreeMap<String, Value>,
    pub active: Option<String>,
}

pub struct Worker {
    pub directory: PathBuf,
    pub catalog: BTreeMap<String, String>,
    pub protocols: Vec<String>,
    pub session_id: String,
    pub max_retained: usize,
    pub max_storage_mb: u64,
    pub max_input_mb: u64,
    pub transport: Value,
    inner: Mutex<State>,
}

fn private_dir(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(path)
}

fn fsync_write(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(data)?;
    file.flush()?;
    file.sync_all()
}

impl Worker {
    pub fn new(directory: PathBuf, backend: Backend, catalog: BTreeMap<String, String>, max_retained: usize, max_storage_mb: u64, max_input_mb: u64, transport: Value) -> std::io::Result<Worker> {
        private_dir(&directory.join("inputs"))?;
        let directory = directory.canonicalize()?;
        Ok(Worker {
            directory,
            catalog,
            protocols: c::SUPPORTED_PROTOCOLS.iter().map(|s| s.to_string()).collect(),
            session_id: crate::sys::uuid4(),
            max_retained,
            max_storage_mb,
            max_input_mb,
            transport,
            inner: Mutex::new(State {
                backend,
                state: "ready".into(),
                loaded: None,
                executions: BTreeMap::new(),
                attempts: BTreeMap::new(),
                operations: BTreeMap::new(),
                inputs: BTreeMap::new(),
                active: None,
            }),
        })
    }

    pub fn lock(&self) -> MutexGuard<'_, State> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn capabilities_of(&self, s: &State) -> Value {
        let mut caps = s.backend.capabilities();
        if let Value::Object(m) = &mut caps {
            m.insert("session_id".into(), Value::str(self.session_id.as_str()));
            m.insert("output_formats".into(), strings(&["png"]));
            m.insert("input_formats".into(), strings(&["image/png"]));
            m.insert("progress".into(), Value::Bool(true));
            m.insert("cancellation".into(), Value::Bool(true));
            m.insert("retention".into(), Value::str("until-discard-or-session-end"));
            m.insert("max_retained_executions".into(), Value::from(self.max_retained));
            m.insert("max_storage_mb".into(), Value::from(self.max_storage_mb));
            m.insert("max_input_mb".into(), Value::from(self.max_input_mb));
            m.insert("required_roles".into(), strings(&c::ROLES));
        }
        caps
    }

    pub fn capabilities(&self) -> Value {
        let s = self.lock();
        self.capabilities_of(&s)
    }

    pub fn info(&self) -> Value {
        let s = self.lock();
        let caps = self.capabilities_of(&s);
        let mut info = obj! {
            "protocol" => self.protocols.last().unwrap().as_str(),
            "supported_protocols" => strings(&self.protocols),
            "worker_version" => c::WORKER_VERSION,
            "backend_id" => BACKEND_ID,
            "ownership" => OWNERSHIP,
            "session_id" => self.session_id.as_str(),
            "state" => s.state.as_str(),
            "service_ready" => s.state != "error",
            "device_ready" => caps["ready"].clone(),
            "model_ready" => s.loaded.is_some(),
            "loaded_bundle" => s.loaded.clone().unwrap_or(Value::Null),
            "device" => s.backend.device.as_str(),
            "build" => obj! {"package" => "media-engine-native", "version" => c::WORKER_VERSION},
            "device_identity" => s.backend.device_identity(),
        };
        if let (Value::Object(m), Value::Object(t)) = (&mut info, &self.transport) {
            m.extend(t.clone());
        }
        info
    }

    pub fn device_identity(&self) -> Value {
        self.lock().backend.device_identity()
    }

    // ----- loading -----------------------------------------------------------------------------

    /// `validate_load()`; returns the root, adapters and the plan.
    fn validate_load(&self, body: &Value, verify: bool) -> Result<(PathBuf, Vec<c::ResolvedAdapter>, Plan)> {
        let map = strict(body, &["protocol", "bundle", "options"], &["adapters", "authorized_fallbacks"])?;
        let version = c::protocol(body, &self.protocols)?;
        let level = c::minor(&version);
        let family = map["bundle"].get("family").and_then(Value::as_str);
        if let Some(name) = family {
            if c::ALL_FAMILIES.iter().any(|(n, since)| *n == name && *since <= level) && name != c::FAMILY {
                // A family this engine does not implement, refused before any file is read.
                return Err(bad("unsupported", "family"));
            }
        }
        if map.contains_key("authorized_fallbacks") && level < 2 {
            return Err(bad("unsupported", "authorized_fallbacks"));
        }
        // quantization and compile exist for 1.3 bodies only; this engine offers only "none".
        let new_options: &[&str] = if level >= 3 { &["quantization", "compile"] } else { &[] };
        let options = strict(&map["options"], &["precision", "attention", "offload"], new_options)?;
        let (precisions, caps) = {
            let s = self.lock();
            (s.backend.precisions(), self.capabilities_of(&s))
        };
        let precision = options["precision"].as_str().unwrap_or("");
        let precision_ok = precisions.contains(&precision) || (level >= 2 && precision == "auto");
        if !precision_ok
            || options["attention"].as_str() != Some("sdpa")
            || options["offload"].as_str() != Some("none")
            || new_options.iter().any(|k| options.get(*k).is_some_and(|v| v.as_str() != Some("none")))
        {
            return Err(bad("unsupported", "options"));
        }
        let authorized: Vec<String> = match map.get("authorized_fallbacks") {
            None => vec![],
            Some(Value::Array(items)) => {
                let fallbacks: Vec<&str> = caps["fallbacks"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
                let list: Vec<String> = items.iter().filter_map(|v| v.as_str().map(str::to_string)).collect();
                let unique: std::collections::BTreeSet<&String> = list.iter().collect();
                if list.len() != items.len() || list.iter().any(|f| !fallbacks.contains(&f.as_str())) || unique.len() != list.len() {
                    return Err(bad("invalid_request", "authorized_fallbacks"));
                }
                list
            }
            Some(_) => return Err(bad("invalid_request", "authorized_fallbacks")),
        };
        let root = c::bundle_path(&map["bundle"], &self.catalog, verify, &version)?;
        let empty = Value::Array(vec![]);
        let adapters = c::validate_adapters(map.get("adapters").unwrap_or(&empty), &self.catalog, &version, verify)?;
        let plan = self.lock().backend.plan(&root, options, &adapters, &authorized)?;
        Ok((root, adapters, plan))
    }

    fn estimate(&self, root: &Path, plan: &Plan, adapters: &[c::ResolvedAdapter], level: u32) -> Result<Value> {
        let mut estimate = {
            let s = self.lock();
            s.backend.compute_estimate(root, plan.options["precision"].as_str().unwrap(), adapters, &plan.device)?
        };
        if level >= 2 {
            if let Value::Object(m) = &mut estimate {
                m.insert("count".into(), obj! {
                    "limit" => vec![Value::Int(1), Value::Int(4)],
                    "scaling" => "activation working memory grows linearly with count; activations are excluded from value_mb",
                });
            }
        }
        self.lock().backend.loaded_estimate = Some(estimate.clone());
        Ok(estimate)
    }

    fn adapter_reports(adapters: &[c::ResolvedAdapter]) -> Value {
        Value::Array(adapters.iter().map(|a| obj! {"adapter_id" => a.record["adapter_id"].clone(), "layout" => a.layout.clone()}).collect())
    }

    /// The /v1/validate response for a load body.
    pub fn validation(&self, body: &Value) -> Result<Value> {
        let (root, adapters, plan) = self.validate_load(body, true)?;
        let level = c::minor(body["protocol"].as_str().unwrap());
        let estimate = self.estimate(&root, &plan, &adapters, level)?;
        let mut result = obj! {"valid" => true, "memory_estimate" => estimate, "adapters" => Self::adapter_reports(&adapters)};
        if level >= 2 {
            if let Value::Object(m) = &mut result {
                m.insert("family".into(), Value::str(c::FAMILY));
                m.insert("tasks".into(), strings(&c::TASKS));
                m.insert("resolved_options".into(), Value::Object(plan.options.clone()));
                m.insert("resolutions".into(), Value::Array(plan.resolutions.clone()));
                m.insert("applied_fallbacks".into(), Value::Array(plan.applied.clone()));
            }
        }
        Ok(result)
    }

    pub fn validate_adapters_only(&self, body: &Value) -> Result<Value> {
        strict(body, &["protocol", "adapters"], &[])?;
        let version = c::protocol(body, &self.protocols)?;
        let adapters = c::validate_adapters(&body["adapters"], &self.catalog, &version, true)?;
        Ok(obj! {"valid" => true, "adapters" => Self::adapter_reports(&adapters)})
    }

    pub fn control(self: &Arc<Self>, action: &str, body: Value) -> Result<Value> {
        if action == "load" {
            let (_root, _adapters, plan) = self.validate_load(&body, false)?;
            let s = self.lock();
            if !(s.backend.device_available() || plan.device == "cpu") {
                return Err(fail(503, "backend_unavailable", None));
            }
        } else {
            strict(&body, &["protocol"], &[])?;
            c::protocol(&body, &self.protocols)?;
        }
        let mut s = self.lock();
        if s.active.is_some() || ["loading", "unloading", "error"].contains(&s.state.as_str()) {
            return Err(fail(409, "busy", None));
        }
        if s.operations.len() >= self.max_retained {
            return Err(fail(503, "storage_full", None));
        }
        let operation_id = format!("op_{}", crate::sys::random_hex());
        let operation = obj! {"operation_id" => operation_id.as_str(), "state" => "running", "action" => action, "error" => Value::Null};
        s.operations.insert(operation_id.clone(), operation.clone());
        s.state = if action == "load" { "loading".into() } else { "unloading".into() };
        drop(s);
        let worker = self.clone();
        let action = action.to_string();
        std::thread::spawn(move || worker.run_control(&operation_id, &action, &body));
        Ok(operation)
    }

    fn unload_backend(&self) -> Value {
        let mut s = self.lock();
        s.backend.pipeline = None;
        s.backend.run_device = s.backend.device.clone();
        s.backend.applied_fallbacks.clear();
        s.loaded = None;
        obj! {"allocated_mb" => Value::Null, "reserved_mb" => Value::Null}
    }

    fn update_operation(&self, s: &mut State, operation_id: &str, fields: Vec<(&str, Value)>) {
        if let Some(Value::Object(op)) = s.operations.get_mut(operation_id) {
            for (k, v) in fields {
                op.insert(k.to_string(), v);
            }
        }
    }

    fn run_control(&self, operation_id: &str, action: &str, body: &Value) {
        let outcome = (|| -> std::result::Result<(), WorkerError> {
            let measurements = self.unload_backend();
            {
                let mut s = self.lock();
                self.update_operation(&mut s, operation_id, vec![("measurements", measurements)]);
            }
            if action != "load" {
                return Ok(());
            }
            let level = c::minor(body["protocol"].as_str().unwrap());
            let (root, adapters, plan) = self.validate_load(body, true)?;
            let estimate = self.estimate(&root, &plan, &adapters, level)?;
            let precision = match plan.options["precision"].as_str() {
                Some("float16") => sdxl::Precision::Float16,
                Some("bfloat16") => sdxl::Precision::BFloat16,
                _ => sdxl::Precision::Float32,
            };
            let device = if plan.device == "cpu" {
                sdxl::DeviceSpec::Cpu
            } else {
                sdxl::DeviceSpec::Cuda(plan.device[5..].parse().unwrap_or(0))
            };
            let specs: Vec<sdxl::Adapter> = adapters
                .iter()
                .map(|a| sdxl::Adapter { weights: a.weights.clone(), scale: a.scale.as_number().unwrap_or(0.0) })
                .collect();
            let pipeline = sdxl::Pipeline::load(&root, precision, device, &specs).map_err(|e| match e {
                sdxl::Error::Unsupported(field) => {
                    eprintln!("bundle configuration outside the native engine's support: {}", field);
                    fail(400, "unsupported", Some("configuration"))
                }
                sdxl::Error::Failed(message) => {
                    eprintln!("load failed: {}", message);
                    fail(500, "model_load_failed", None)
                }
                _ => fail(500, "model_load_failed", None),
            })?;
            let mut s = self.lock();
            s.backend.pipeline = Some(Arc::new(pipeline));
            s.backend.run_device = plan.device.clone();
            s.backend.applied_fallbacks = plan.applied.clone();
            let adapters_loaded: Vec<Value> = adapters
                .iter()
                .map(|a| obj! {"adapter_id" => a.record["adapter_id"].clone(), "digest" => a.record["digest"].clone(), "scale" => a.scale.clone(), "layout" => a.layout.clone()})
                .collect();
            let mut loaded = obj! {
                "bundle_digest" => body["bundle"]["bundle_digest"].clone(),
                "bundle" => body["bundle"].clone(),
                "options" => body["options"].clone(),
                "adapters" => Value::Array(adapters_loaded),
                "memory_estimate" => estimate.clone(),
                "family" => c::FAMILY,
                "tasks" => strings(&c::TASKS),
                "resolved_options" => Value::Object(plan.options.clone()),
                "applied_fallbacks" => Value::Array(plan.applied.clone()),
            };
            let _ = &mut loaded;
            s.loaded = Some(loaded);
            let mut fields = vec![("memory_estimate", estimate)];
            if level >= 2 {
                fields.push(("resolved_options", Value::Object(plan.options.clone())));
                fields.push(("resolutions", Value::Array(plan.resolutions.clone())));
                fields.push(("applied_fallbacks", Value::Array(plan.applied.clone())));
            }
            self.update_operation(&mut s, operation_id, fields);
            Ok(())
        })();
        if let Err(e) = &outcome {
            self.unload_backend();
            let mut s = self.lock();
            self.update_operation(&mut s, operation_id, vec![("state", Value::str("failed")), ("error", error_body(e.code, e.field.as_deref()))]);
            s.state = "ready".into();
            return;
        }
        let mut s = self.lock();
        self.update_operation(&mut s, operation_id, vec![("state", Value::str("succeeded"))]);
        s.state = "ready".into();
    }

    // ----- inputs ------------------------------------------------------------------------------

    fn storage_used(&self) -> u64 {
        let mut total = 0;
        if let Ok(entries) = std::fs::read_dir(&self.directory) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("exec_") {
                    if let Ok(files) = std::fs::read_dir(entry.path()) {
                        for f in files.flatten() {
                            if f.path().extension().is_some_and(|e| e == "png") {
                                total += f.metadata().map(|m| m.len()).unwrap_or(0);
                            }
                        }
                    }
                }
            }
        }
        if let Ok(files) = std::fs::read_dir(self.directory.join("inputs")) {
            for f in files.flatten() {
                if f.path().extension().is_some_and(|e| e == "png") {
                    total += f.metadata().map(|m| m.len()).unwrap_or(0);
                }
            }
        }
        total
    }

    pub fn store_input(&self, data: &[u8]) -> Result<Value> {
        let (width, height) = c::png_dimensions(data)?;
        let mut s = self.lock();
        if s.state == "error" {
            return Err(fail(409, "busy", None));
        }
        if s.inputs.len() + s.executions.len() >= self.max_retained || self.storage_used() + data.len() as u64 > self.max_storage_mb * 1024 * 1024 {
            return Err(fail(503, "storage_full", None));
        }
        let input_id = format!("input_{}", crate::sys::random_hex());
        let path = self.directory.join("inputs").join(format!("{}.png", input_id));
        let temporary = path.with_extension("part");
        fsync_write(&temporary, data).map_err(|_| fail(500, "internal_error", None))?;
        std::fs::rename(&temporary, &path).map_err(|_| fail(500, "internal_error", None))?;
        let record = obj! {
            "input_id" => input_id.as_str(),
            "session_id" => self.session_id.as_str(),
            "mime_type" => "image/png",
            "size" => data.len(),
            "sha256" => crate::sha256::hex_digest(data),
            "width" => width as u64,
            "height" => height as u64,
        };
        s.inputs.insert(input_id, record.clone());
        Ok(record)
    }

    pub fn input_record(&self, input_id: &str) -> Result<Value> {
        self.lock().inputs.get(input_id).cloned().ok_or_else(|| fail(404, "input_not_found", None))
    }

    pub fn release_input(&self, input_id: &str) -> Result<()> {
        let mut s = self.lock();
        if !s.inputs.contains_key(input_id) {
            return Err(fail(404, "input_not_found", None));
        }
        let in_use = s.executions.values().any(|e| {
            e.request["inputs"].get("image").and_then(Value::as_str) == Some(input_id) && !TERMINAL.contains(&e.state.as_str())
        });
        if in_use {
            return Err(fail(409, "busy", None));
        }
        let _ = std::fs::remove_file(self.directory.join("inputs").join(format!("{}.png", input_id)));
        s.inputs.remove(input_id);
        Ok(())
    }

    // ----- executions --------------------------------------------------------------------------

    fn loaded_checks(&self, s: &State, body: &Value, task: &str) -> Result<()> {
        if let Some(loaded) = &s.loaded {
            if loaded["bundle_digest"] == body["bundle_digest"] && !c::TASKS.contains(&task) {
                return Err(bad("unsupported", "task"));
            }
        }
        Ok(())
    }

    pub fn validate_execution(&self, body: &Value) -> Result<Value> {
        let task = c::validate_parameters(body, &self.protocols)?;
        let level = c::minor(body["protocol"].as_str().unwrap());
        {
            let s = self.lock();
            self.loaded_checks(&s, body, &task)?;
        }
        let params = body["parameters"].as_object().unwrap();
        let mut result = obj! {"valid" => true, "task" => task.as_str(), "effective_steps" => c::effective_steps(params, &task)};
        if level >= 2 {
            if let Value::Object(m) = &mut result {
                m.insert("outputs".into(), strings(&c::output_ids(&task, params)));
                m.insert("effective_settings".into(), c::effective_settings(params, &task, body["protocol"].as_str().unwrap()));
            }
        }
        Ok(result)
    }

    fn snapshot_of(&self, e: &Execution) -> Value {
        let mut snap = obj! {
            "attempt_id" => e.attempt_id.as_str(),
            "execution_id" => e.execution_id.as_str(),
            "session_id" => self.session_id.as_str(),
            "protocol" => e.protocol.as_str(),
            "task" => e.task.as_str(),
            "state" => e.state.as_str(),
            "stage" => e.stage.as_str(),
            "progress" => e.progress.clone(),
            "events" => Value::Array(e.events.clone()),
            "last_sequence" => e.events.len(),
            "effective_settings" => e.effective_settings.clone(),
            "bundle_digest" => e.bundle_digest.as_str(),
            "implementation" => e.implementation.clone(),
            "cancel_requested" => e.cancel_requested,
            "outputs" => Value::Array(e.outputs.clone()),
            "error" => e.error.clone(),
        };
        if let Value::Object(m) = &mut snap {
            if let Some(f) = &e.applied_fallbacks {
                m.insert("applied_fallbacks".into(), f.clone());
            }
            if let Some(ms) = &e.measurements {
                m.insert("measurements".into(), ms.clone());
            }
        }
        snap
    }

    pub fn snapshot(&self, execution_id: &str) -> Result<Value> {
        let s = self.lock();
        s.executions.get(execution_id).map(|e| self.snapshot_of(e)).ok_or_else(|| fail(404, "execution_not_found", None))
    }

    /// Events after `cursor`, the last sequence and whether the execution is terminal.
    pub fn events_after(&self, execution_id: &str, cursor: usize) -> Result<(Vec<Value>, usize, bool)> {
        let s = self.lock();
        let e = s.executions.get(execution_id).ok_or_else(|| fail(404, "execution_not_found", None))?;
        let events = e.events.get(cursor..).map(<[Value]>::to_vec).unwrap_or_default();
        Ok((events, e.events.len(), TERMINAL.contains(&e.state.as_str())))
    }

    pub fn attempt(&self, attempt_id: &str) -> Result<Value> {
        let s = self.lock();
        let id = s.attempts.get(attempt_id).ok_or_else(|| fail(404, "execution_not_found", None))?;
        Ok(self.snapshot_of(&s.executions[id]))
    }

    fn emit(&self, s: &mut State, execution_id: &str, kind: &str, data: Value) {
        let Some(e) = s.executions.get_mut(execution_id) else { return };
        let event = obj! {
            "session_id" => self.session_id.as_str(),
            "execution_id" => execution_id,
            "sequence" => e.events.len() + 1,
            "timestamp" => crate::sys::utc_now(),
            "type" => kind,
            "data" => data.clone(),
        };
        e.events.push(event);
        if kind == "stage" {
            e.stage = data["stage"].as_str().unwrap_or("").to_string();
        }
        if kind == "progress" {
            e.progress = data;
        }
    }

    pub fn submit(self: &Arc<Self>, body: Value) -> Result<Value> {
        let task = c::validate_parameters(&body, &self.protocols)?;
        let attempt = body["attempt_id"].as_str().unwrap().to_string();
        let level = c::minor(body["protocol"].as_str().unwrap());
        let mut s = self.lock();
        if let Some(id) = s.attempts.get(&attempt) {
            let entry = &s.executions[id];
            if entry.request != body {
                return Err(fail(409, "attempt_conflict", None));
            }
            return Ok(self.snapshot_of(entry));
        }
        self.loaded_checks(&s, &body, &task)?;
        if s.active.is_some() || s.state != "ready" {
            return Err(fail(409, "busy", None));
        }
        if !s.loaded.as_ref().is_some_and(|l| l["bundle_digest"] == body["bundle_digest"]) {
            return Err(fail(409, "model_not_ready", None));
        }
        let params = body["parameters"].as_object().unwrap().clone();
        let mut image_path = None;
        if task == "image-to-image" {
            let id = body["inputs"]["image"].as_str().unwrap();
            let record = s.inputs.get(id).ok_or_else(|| fail(404, "input_not_found", None))?;
            if record["width"] != params["width"] || record["height"] != params["height"] {
                return Err(bad("invalid_request", "width"));
            }
            image_path = Some(self.directory.join("inputs").join(format!("{}.png", id)));
        }
        let expected = c::output_ids(&task, &params);
        let width = params["width"].as_int().unwrap() as u64;
        let height = params["height"].as_int().unwrap() as u64;
        let reserve = (width * height * 4 + 65536) * expected.len() as u64;
        if s.executions.len() + s.inputs.len() >= self.max_retained || self.storage_used() + reserve > self.max_storage_mb * 1024 * 1024 {
            return Err(fail(503, "storage_full", None));
        }
        let execution_id = format!("exec_{}", crate::sys::random_hex());
        s.active = Some(execution_id.clone());
        s.state = "busy".into();
        s.attempts.insert(attempt.clone(), execution_id.clone());
        let fallbacks = s.backend.applied_fallbacks.clone();
        let entry = Execution {
            attempt_id: attempt,
            execution_id: execution_id.clone(),
            protocol: body["protocol"].as_str().unwrap().to_string(),
            task: task.clone(),
            state: "running".into(),
            stage: "encoding".into(),
            progress: Value::Null,
            events: Vec::new(),
            effective_settings: c::effective_settings(&params, &task, body["protocol"].as_str().unwrap()),
            bundle_digest: body["bundle_digest"].as_str().unwrap().to_string(),
            implementation: self.capabilities_of(&s),
            cancel_requested: false,
            outputs: Vec::new(),
            error: Value::Null,
            applied_fallbacks: if level >= 2 || !fallbacks.is_empty() { Some(Value::Array(fallbacks)) } else { None },
            measurements: None,
            request: body,
            cancel: Arc::new(AtomicBool::new(false)),
            image_path,
            expected_outputs: expected,
        };
        s.executions.insert(execution_id.clone(), entry);
        self.emit(&mut s, &execution_id, "state", obj! {"state" => "running"});
        let snapshot = self.snapshot_of(&s.executions[&execution_id]);
        drop(s);
        let worker = self.clone();
        let id = execution_id.clone();
        std::thread::spawn(move || worker.run(&id));
        Ok(snapshot)
    }

    fn run(&self, execution_id: &str) {
        let folder = self.directory.join(execution_id);
        let temporary = folder.join(".partial");
        let (pipeline, request, cancel, image_path, expected) = {
            let s = self.lock();
            let e = &s.executions[execution_id];
            (s.backend.pipeline.clone(), e.request.clone(), e.cancel.clone(), e.image_path.clone(), e.expected_outputs.clone())
        };
        enum Outcome {
            Succeeded(Vec<Value>, Value),
            Cancelled,
            Failed(Value),
        }
        let outcome = (|| -> Outcome {
            let Some(pipeline) = pipeline else { return Outcome::Failed(error_body("execution_failed", None)) };
            if private_dir(&temporary).is_err() {
                return Outcome::Failed(error_body("execution_failed", None));
            }
            let params = request["parameters"].as_object().unwrap();
            let task = request["task"].as_str().unwrap();
            let image_png = match &image_path {
                Some(p) => match std::fs::read(p) {
                    Ok(d) => Some(d),
                    Err(_) => return Outcome::Failed(error_body("execution_failed", None)),
                },
                None => None,
            };
            let spec = sdxl::Request {
                task: if task == "image-to-image" { sdxl::Task::ImageToImage } else { sdxl::Task::TextToImage },
                prompt: request["inputs"]["prompt"].as_str().unwrap().to_string(),
                negative_prompt: request["inputs"].get("negative_prompt").and_then(Value::as_str).map(str::to_string),
                seed: params["seed"].as_int().unwrap() as u64,
                steps: params["steps"].as_int().unwrap() as usize,
                guidance: params["guidance"].as_number().unwrap(),
                width: params["width"].as_int().unwrap() as usize,
                height: params["height"].as_int().unwrap() as usize,
                count: params.get("count").and_then(Value::as_int).unwrap_or(1) as usize,
                strength: params.get("strength").and_then(Value::as_number).unwrap_or(1.0),
                image_png,
            };
            let mut events = |event: sdxl::Event| {
                let mut s = self.lock();
                match event {
                    sdxl::Event::Stage(stage) => self.emit(&mut s, execution_id, "stage", obj! {"stage" => stage}),
                    sdxl::Event::Progress { completed, total } => self.emit(
                        &mut s,
                        execution_id,
                        "progress",
                        obj! {"stage" => "sampling", "completed_steps" => completed, "total_steps" => total, "scope" => "stage"},
                    ),
                }
            };
            let cancelled = || cancel.load(Ordering::SeqCst);
            let output = match pipeline.generate(&spec, &mut events, &cancelled) {
                Ok(o) => o,
                Err(sdxl::Error::Cancelled) => return Outcome::Cancelled,
                Err(sdxl::Error::Failed(message)) => {
                    let code = if message.to_ascii_lowercase().contains("out of memory") { "out_of_memory" } else { "execution_failed" };
                    return Outcome::Failed(error_body(code, None));
                }
                Err(_) => return Outcome::Failed(error_body("execution_failed", None)),
            };
            if cancelled() {
                return Outcome::Cancelled;
            }
            {
                let mut s = self.lock();
                self.emit(&mut s, execution_id, "stage", obj! {"stage" => "saving"});
            }
            for (output_id, data) in expected.iter().zip(&output.images) {
                if fsync_write(&temporary.join(format!("{}.png", output_id)), data).is_err() {
                    return Outcome::Failed(error_body("execution_failed", None));
                }
            }
            // Verify every advertised output, read its facts back from the file, then publish.
            let mut records = Vec::new();
            for output_id in &expected {
                let path = temporary.join(format!("{}.png", output_id));
                let Ok(data) = std::fs::read(&path) else {
                    return Outcome::Failed(error_body("execution_failed", Some(output_id)));
                };
                let Ok((width, height)) = png_header(&data) else {
                    return Outcome::Failed(error_body("execution_failed", Some(output_id)));
                };
                let final_path = folder.join(format!("{}.png", output_id));
                if std::fs::rename(&path, &final_path).is_err() {
                    return Outcome::Failed(error_body("execution_failed", None));
                }
                records.push(obj! {
                    "output_id" => output_id.as_str(),
                    "mime_type" => "image/png",
                    "size" => data.len(),
                    "sha256" => crate::sha256::hex_digest(&data),
                    "width" => width as u64,
                    "height" => height as u64,
                });
            }
            let measurements = obj! {
                "effective_steps" => output.effective_steps,
                "peak_allocated_mb" => Value::Null,
                "peak_reserved_mb" => Value::Null,
                "process_peak_rss_mb" => crate::sys::peak_rss_mb(),
            };
            Outcome::Succeeded(records, measurements)
        })();
        let _ = std::fs::remove_dir_all(&temporary);
        let mut s = self.lock();
        // Decided under the lock cancel() takes: a cancel recorded before publication wins, so
        // `cancel_requested` is true exactly when the execution does not succeed.
        let requested = s.executions.get(execution_id).is_some_and(|e| e.cancel_requested);
        let outcome = match outcome {
            Outcome::Succeeded(..) if requested => Outcome::Cancelled,
            other => other,
        };
        let state = match outcome {
            Outcome::Succeeded(records, measurements) => {
                if let Some(e) = s.executions.get_mut(execution_id) {
                    e.outputs = records.clone();
                    e.measurements = Some(measurements);
                }
                for record in records {
                    self.emit(&mut s, execution_id, "output", record);
                }
                "succeeded"
            }
            Outcome::Cancelled => {
                let _ = std::fs::remove_dir_all(&folder);
                "cancelled"
            }
            Outcome::Failed(failure) => {
                let _ = std::fs::remove_dir_all(&folder);
                if let Some(e) = s.executions.get_mut(execution_id) {
                    e.error = failure.clone();
                }
                let oom = failure["code"].as_str() == Some("out_of_memory");
                self.emit(&mut s, execution_id, "error", failure);
                if oom {
                    s.state = "error".into();
                }
                "failed"
            }
        };
        if let Some(e) = s.executions.get_mut(execution_id) {
            e.state = state.into();
        }
        self.emit(&mut s, execution_id, "state", obj! {"state" => state});
        s.active = None;
        if s.state != "error" {
            s.state = "ready".into();
        }
    }

    pub fn cancel(&self, execution_id: &str) -> Result<Value> {
        let mut s = self.lock();
        let e = s.executions.get_mut(execution_id).ok_or_else(|| fail(404, "execution_not_found", None))?;
        if !TERMINAL.contains(&e.state.as_str()) {
            e.cancel_requested = true;
            e.cancel.store(true, Ordering::SeqCst);
        }
        Ok(self.snapshot_of(&s.executions[execution_id]))
    }

    /// Stop admission and cancel the running execution (SIGTERM, SIGINT, parent exit).
    pub fn stop(&self) {
        let mut s = self.lock();
        s.state = "error".into();
        if let Some(id) = s.active.clone() {
            if let Some(e) = s.executions.get_mut(&id) {
                e.cancel_requested = true;
                e.cancel.store(true, Ordering::SeqCst);
            }
        }
    }

    pub fn is_active(&self) -> bool {
        self.lock().active.is_some()
    }

    pub fn output_path(&self, execution_id: &str, output_id: &str) -> Result<(PathBuf, String)> {
        let s = self.lock();
        let e = s.executions.get(execution_id).ok_or_else(|| fail(404, "execution_not_found", None))?;
        if e.state != "succeeded" {
            return Err(fail(404, "output_not_found", None));
        }
        let record = e.outputs.iter().find(|o| o["output_id"].as_str() == Some(output_id)).ok_or_else(|| fail(404, "output_not_found", None))?;
        Ok((self.directory.join(execution_id).join(format!("{}.png", output_id)), record["mime_type"].as_str().unwrap().to_string()))
    }

    pub fn release(&self, execution_id: &str) -> Result<()> {
        let mut s = self.lock();
        let e = s.executions.get(execution_id).ok_or_else(|| fail(404, "execution_not_found", None))?;
        if !TERMINAL.contains(&e.state.as_str()) {
            return Err(fail(409, "busy", None));
        }
        let attempt = e.attempt_id.clone();
        let _ = std::fs::remove_dir_all(self.directory.join(execution_id));
        s.attempts.remove(&attempt);
        s.executions.remove(execution_id);
        Ok(())
    }

    pub fn is_error(&self) -> bool {
        self.lock().state == "error"
    }
}

fn png_header(data: &[u8]) -> std::result::Result<(u32, u32), ()> {
    if data.len() < 33 || &data[..8] != b"\x89PNG\r\n\x1a\n" || &data[12..16] != b"IHDR" {
        return Err(());
    }
    Ok((u32::from_be_bytes(data[16..20].try_into().unwrap()), u32::from_be_bytes(data[20..24].try_into().unwrap())))
}
