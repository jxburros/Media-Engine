//! Owned HTTP/1.1 server: one thread per connection, one request per connection (the response
//! closes it, as Python's `BaseHTTPRequestHandler` does), Content-Length bodies, SSE streaming.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

pub struct Request {
    pub method: String,
    pub path: String,
    pub query: String,
    headers: HashMap<String, String>,
    pub reader: BufReader<TcpStream>,
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
    pub stream: &'a mut TcpStream,
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

fn read_request(stream: &TcpStream) -> Option<Request> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);
    let mut line = String::new();
    let mut total = 0usize;
    reader.read_line(&mut line).ok()?;
    let mut parts = line.trim_end().split(' ');
    let method = parts.next()?.to_string();
    let target = parts.next()?.to_string();
    if !parts.next()?.starts_with("HTTP/1.") {
        return None;
    }
    let mut headers = HashMap::new();
    loop {
        let mut h = String::new();
        let n = reader.read_line(&mut h).ok()?;
        total += n;
        if n == 0 || total > 64 * 1024 {
            return None;
        }
        let h = h.trim_end_matches(['\r', '\n']);
        if h.is_empty() {
            break;
        }
        let (k, v) = h.split_once(':')?;
        headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
    }
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target, String::new()),
    };
    Some(Request { method, path, query, headers, reader })
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
pub fn serve<F>(listener: TcpListener, handler: F)
where
    F: Fn(Request, Response) + Send + Sync + 'static,
{
    let handler = std::sync::Arc::new(handler);
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let handler = handler.clone();
        std::thread::spawn(move || {
            let _ = stream.set_read_timeout(Some(Duration::from_secs(15)));
            if let Some(request) = read_request(&stream) {
                let response = Response { stream: &mut stream };
                handler(request, response);
            }
            let _ = stream.shutdown(std::net::Shutdown::Both);
        });
    }
}
