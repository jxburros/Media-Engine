//! Owned HTTP/1.1 server: one thread per connection, one request per connection (the response
//! closes it, as Python's `BaseHTTPRequestHandler` does), Content-Length bodies, SSE streaming.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;
use std::sync::{Arc, Mutex};

pub const MAX_JSON_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_HEADER_BYTES: usize = 64 * 1024;
pub const MAX_HEADER_COUNT: usize = 64;
pub const MAX_REQUEST_LINE_BYTES: usize = 8192;

enum Stream { Plain(TcpStream), Tls(rustls::StreamOwned<rustls::ServerConnection, TcpStream>) }

#[derive(Clone)]
pub struct Connection(Arc<Mutex<Stream>>);

impl Read for Connection {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        match &mut *self.0.lock().unwrap() { Stream::Plain(s) => s.read(bytes), Stream::Tls(s) => s.read(bytes) }
    }
}
impl Write for Connection {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        match &mut *self.0.lock().unwrap() { Stream::Plain(s) => s.write(bytes), Stream::Tls(s) => s.write(bytes) }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match &mut *self.0.lock().unwrap() { Stream::Plain(s) => s.flush(), Stream::Tls(s) => s.flush() }
    }
}

pub struct Request {
    pub method: String,
    pub path: String,
    pub query: String,
    headers: HashMap<String, String>,
    pub reader: BufReader<Connection>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(&name.to_ascii_lowercase()).map(String::as_str)
    }
    /// The body, which must declare 1..=limit bytes; `Err(413)` above the limit, `Err(400)`
    /// for none or a malformed length.
    pub fn body(&mut self, limit: usize) -> Result<Vec<u8>, u16> {
        let size: usize = match self.header("content-length") {
            None => 0,
            Some(v) => v.trim().parse().map_err(|_| 400u16)?,
        };
        if size == 0 {
            return Err(400);
        }
        if size > limit {
            return Err(413);
        }
        let mut data = vec![0u8; size];
        self.reader.read_exact(&mut data).map_err(|_| 400u16)?;
        Ok(data)
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        410 => "Gone",
        413 => "Request Entity Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "",
    }
}

pub struct Response<'a> {
    pub stream: &'a mut Connection,
}

impl<'a> Response<'a> {
    pub fn head(&mut self, status: u16, headers: &[(&str, String)]) -> std::io::Result<()> {
        let mut text = format!("HTTP/1.1 {} {}\r\nServer: media-engine-native\r\nConnection: close\r\n", status, reason(status));
        for (k, v) in headers {
            text.push_str(&format!("{}: {}\r\n", k, v));
        }
        text.push_str("\r\n");
        self.stream.write_all(text.as_bytes())
    }
    pub fn send(&mut self, status: u16, content_type: &str, body: &[u8]) -> std::io::Result<()> {
        self.head(
            status,
            &[
                ("Content-Type", content_type.to_string()),
                ("Content-Length", body.len().to_string()),
                ("Cache-Control", "no-store".to_string()),
            ],
        )?;
        self.stream.write_all(body)?;
        self.stream.flush()
    }
    pub fn write(&mut self, data: &[u8]) -> std::io::Result<()> {
        self.stream.write_all(data)?;
        self.stream.flush()
    }
}

fn read_line(reader: &mut BufReader<Connection>, limit: usize) -> Option<String> {
    let mut bytes = Vec::new();
    reader.take((limit + 1) as u64).read_until(b'\n', &mut bytes).ok()?;
    if bytes.len() > limit || !bytes.ends_with(b"\r\n") { return None; }
    String::from_utf8(bytes).ok()
}

fn read_request(stream: &Connection) -> Option<Request> {
    let mut reader = BufReader::new(stream.clone());
    let line = read_line(&mut reader, MAX_REQUEST_LINE_BYTES)?;
    let mut parts = line.trim_end().split(' ');
    let method = parts.next()?.to_string();
    let target = parts.next()?.to_string();
    if !matches!(parts.next()?, "HTTP/1.0" | "HTTP/1.1") || parts.next().is_some() || !target.starts_with('/') {
        return None;
    }
    let mut headers = HashMap::new();
    let mut total = 0;
    loop {
        let h = read_line(&mut reader, MAX_HEADER_BYTES - total)?;
        total += h.len();
        if total > MAX_HEADER_BYTES { return None; }
        let h = h.trim_end_matches(['\r', '\n']);
        if h.is_empty() { break; }
        if headers.len() >= MAX_HEADER_COUNT { return None; }
        let (k, v) = h.split_once(':')?;
        if k.is_empty() || !k.bytes().all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)) {
            return None;
        }
        let k = k.to_ascii_lowercase();
        // Duplicate headers are unnecessary in this protocol and ambiguity is refused.
        if headers.insert(k, v.trim().to_string()).is_some() { return None; }
    }
    if headers.contains_key("transfer-encoding") { return None; }
    if let Some(value) = headers.get("content-length") {
        if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) { return None; }
    }
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target, String::new()),
    };
    Some(Request { method, path, query, headers, reader })
}

pub fn loopback_host(value: &str) -> bool {
    let host = if value.starts_with('[') {
        let Some(end) = value.find(']') else { return false };
        let suffix = &value[end + 1..];
        if !suffix.is_empty() && !suffix.strip_prefix(':').is_some_and(|p| p.parse::<u16>().is_ok()) { return false; }
        &value[1..end]
    } else {
        match value.split_once(':') {
            Some((host, port)) if port.parse::<u16>().is_ok() => host,
            Some(_) => return false,
            None => value,
        }
    };
    host.eq_ignore_ascii_case("localhost") || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// Percent-decode one query parameter value (`attempt_id=` lookups).
pub fn query_value(query: &str, name: &str) -> Option<String> {
    for pair in query.split('&') {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        if k == name {
            let bytes = v.as_bytes();
            let mut out = Vec::new();
            let mut i = 0;
            while i < bytes.len() {
                match bytes[i] {
                    b'%' if i + 3 <= bytes.len() => {
                        let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok()?;
                        out.push(u8::from_str_radix(hex, 16).ok()?);
                        i += 3;
                    }
                    b'+' => {
                        out.push(b' ');
                        i += 1;
                    }
                    b => {
                        out.push(b);
                        i += 1;
                    }
                }
            }
            return String::from_utf8(out).ok();
        }
    }
    None
}

/// Accept connections forever, handling each on its own thread.
pub fn serve<F>(listener: TcpListener, tls: Option<Arc<rustls::ServerConfig>>, handler: F)
where
    F: Fn(Request, Response) + Send + Sync + 'static,
{
    let handler = std::sync::Arc::new(handler);
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let handler = handler.clone();
        let tls = tls.clone();
        std::thread::spawn(move || {
            let _ = stream.set_read_timeout(Some(Duration::from_secs(15)));
            let _ = stream.set_write_timeout(Some(Duration::from_secs(15)));
            let connection = match tls {
                Some(config) => {
                    let Ok(mut session) = rustls::ServerConnection::new(config) else { return };
                    let mut socket = stream;
                    while session.is_handshaking() {
                        if session.complete_io(&mut socket).is_err() { return; }
                    }
                    Stream::Tls(rustls::StreamOwned::new(session, socket))
                }
                None => Stream::Plain(stream),
            };
            let mut stream = Connection(Arc::new(Mutex::new(connection)));
            if let Some(request) = read_request(&stream) {
                let response = Response { stream: &mut stream };
                handler(request, response);
            } else {
                let _ = Response { stream: &mut stream }.send(400, "application/json", b"{\"error\":{\"code\":\"invalid_request\"}}");
            }

        });
    }
}
