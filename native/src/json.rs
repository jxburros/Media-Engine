//! Owned JSON: a strict parser and two serializers.
//!
//! `Value` keeps integers and floats apart because the protocol does (`64` is a width, `64.0`
//! is not), and object keys sorted so `canonical()` reproduces Python's
//! `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`, which every
//! registration digest is computed over.

use std::collections::BTreeMap;
use std::fmt::Write;

#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

pub type Map = BTreeMap<String, Value>;

impl PartialEq for Value {
    /// Python equality: `1 == 1.0`, and `True` is not compared equal to `1` here because the
    /// protocol never mixes them (bool is never a valid number).
    fn eq(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => (*a as f64) == *b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => a == b,
            (Value::Object(a), Value::Object(b)) => a == b,
            _ => false,
        }
    }
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(map) => map.get(key),
            _ => None,
        }
    }
    pub fn as_object(&self) -> Option<&Map> {
        match self {
            Value::Object(map) => Some(map),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
    /// An integer in the JSON sense (`type(v) is int` in Python; never a bool or a float).
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }
    /// An int or a float (`type(v) in (int, float)`), never a bool.
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }
    pub fn str(s: impl Into<String>) -> Value {
        Value::Str(s.into())
    }
}

static NULL: Value = Value::Null;

/// `value["key"]`: the member, or null when absent or when `value` is not an object.
impl std::ops::Index<&str> for Value {
    type Output = Value;
    fn index(&self, key: &str) -> &Value {
        self.get(key).unwrap_or(&NULL)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Value {
        Value::Str(s.to_string())
    }
}
impl From<String> for Value {
    fn from(s: String) -> Value {
        Value::Str(s)
    }
}
impl From<i64> for Value {
    fn from(i: i64) -> Value {
        Value::Int(i)
    }
}
impl From<usize> for Value {
    fn from(i: usize) -> Value {
        Value::Int(i as i64)
    }
}
impl From<u64> for Value {
    fn from(i: u64) -> Value {
        Value::Int(i as i64)
    }
}
impl From<bool> for Value {
    fn from(b: bool) -> Value {
        Value::Bool(b)
    }
}
impl From<f64> for Value {
    fn from(f: f64) -> Value {
        Value::Float(f)
    }
}
impl From<Map> for Value {
    fn from(m: Map) -> Value {
        Value::Object(m)
    }
}
impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Value {
        Value::Array(v)
    }
}
impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(o: Option<T>) -> Value {
        o.map(Into::into).unwrap_or(Value::Null)
    }
}

/// Build an object from `(key, value)` pairs.
#[macro_export]
macro_rules! obj {
    () => { $crate::json::Value::Object($crate::json::Map::new()) };
    ($($k:expr => $v:expr),+ $(,)?) => {{
        let mut m = $crate::json::Map::new();
        $( m.insert(($k).to_string(), $crate::json::Value::from($v)); )+
        $crate::json::Value::Object(m)
    }};
}

/// Build an array of string values.
pub fn strings<S: AsRef<str>>(items: &[S]) -> Value {
    Value::Array(items.iter().map(|s| Value::str(s.as_ref())).collect())
}

// ----- parsing ---------------------------------------------------------------------------------

pub fn parse(text: &[u8]) -> Result<Value, ()> {
    let text = std::str::from_utf8(text).map_err(|_| ())?;
    let mut p = Parser { s: text.as_bytes(), i: 0, depth: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(());
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }
    fn expect(&mut self, word: &str) -> Result<(), ()> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(())
        } else {
            Err(())
        }
    }
    fn value(&mut self) -> Result<Value, ()> {
        self.depth += 1;
        if self.depth > 512 {
            return Err(());
        }
        let v = match self.peek().ok_or(())? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => self.string().map(Value::Str),
            b't' => self.expect("true").map(|_| Value::Bool(true)),
            b'f' => self.expect("false").map(|_| Value::Bool(false)),
            b'n' => self.expect("null").map(|_| Value::Null),
            _ => self.number(),
        };
        self.depth -= 1;
        v
    }
    fn object(&mut self) -> Result<Value, ()> {
        self.i += 1;
        let mut map = Map::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Value::Object(map));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err(());
            }
            let key = self.string()?;
            self.ws();
            if self.peek() != Some(b':') {
                return Err(());
            }
            self.i += 1;
            self.ws();
            let value = self.value()?;
            map.insert(key, value); // duplicate keys: the last wins, as in Python
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Value::Object(map));
                }
                _ => return Err(()),
            }
        }
    }
    fn array(&mut self) -> Result<Value, ()> {
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(()),
            }
        }
    }
    fn hex4(&mut self) -> Result<u32, ()> {
        let digits = self.s.get(self.i..self.i + 4).ok_or(())?;
        let text = std::str::from_utf8(digits).map_err(|_| ())?;
        let v = u32::from_str_radix(text, 16).map_err(|_| ())?;
        self.i += 4;
        Ok(v)
    }
    fn string(&mut self) -> Result<String, ()> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let c = self.peek().ok_or(())?;
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.i += 1;
                    let e = self.peek().ok_or(())?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut code = self.hex4()?;
                            if (0xD800..0xDC00).contains(&code) && self.s[self.i..].starts_with(b"\\u") {
                                let save = self.i;
                                self.i += 2;
                                let low = self.hex4()?;
                                if (0xDC00..0xE000).contains(&low) {
                                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                                } else {
                                    self.i = save;
                                }
                            }
                            // Lone surrogates cannot be represented in a Rust String.
                            out.push(char::from_u32(code).ok_or(())?);
                        }
                        _ => return Err(()),
                    }
                }
                0..=0x1f => return Err(()),
                _ => {
                    // Copy one UTF-8 scalar.
                    let rest = std::str::from_utf8(&self.s[self.i..]).map_err(|_| ())?;
                    let ch = rest.chars().next().ok_or(())?;
                    out.push(ch);
                    self.i += ch.len_utf8();
                }
            }
        }
    }
    fn number(&mut self) -> Result<Value, ()> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        match self.peek() {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.i += 1;
                }
            }
            _ => return Err(()),
        }
        let mut float = false;
        if self.peek() == Some(b'.') {
            float = true;
            self.i += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(());
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            float = true;
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(());
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).map_err(|_| ())?;
        if !float {
            if let Ok(i) = text.parse::<i64>() {
                return Ok(Value::Int(i));
            }
            // Out of i64 range: Python would keep an exact int, which no field accepts anyway.
            return Ok(Value::Float(text.parse::<f64>().map_err(|_| ())?));
        }
        let f = text.parse::<f64>().map_err(|_| ())?;
        if !f.is_finite() {
            return Err(());
        }
        Ok(Value::Float(f))
    }
}

// ----- serializing -----------------------------------------------------------------------------

/// Python's `repr(float)`: shortest round-trip digits, `1.0` for integral values, exponent form
/// below 1e-4 and from 1e16.
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let exp = f.abs().log10().floor() as i32;
    if (-4..16).contains(&exp) {
        let s = format!("{}", f);
        if s.contains('.') || s.contains('e') {
            s
        } else {
            s + ".0"
        }
    } else {
        // Rust's {:e} gives the shortest digits as d.ddde±x; Python writes e-05 / e+16.
        let s = format!("{:e}", f);
        let (mantissa, exponent) = s.split_once('e').unwrap();
        let value: i32 = exponent.parse().unwrap();
        let sign = if value < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", mantissa, sign, value.abs())
    }
}

fn write_string(out: &mut String, s: &str, ascii: bool) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c if ascii && (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{:04x}", unit);
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_value(out: &mut String, v: &Value, ascii: bool, compact: bool) {
    let (comma, colon) = if compact { (",", ":") } else { (", ", ": ") };
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Int(i) => {
            let _ = write!(out, "{}", i);
        }
        Value::Float(f) => out.push_str(&float_repr(*f)),
        Value::Str(s) => write_string(out, s, ascii),
        Value::Array(items) => {
            out.push('[');
            for (k, item) in items.iter().enumerate() {
                if k > 0 {
                    out.push_str(comma);
                }
                write_value(out, item, ascii, compact);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (k, (key, item)) in map.iter().enumerate() {
                if k > 0 {
                    out.push_str(comma);
                }
                write_string(out, key, ascii);
                out.push_str(colon);
                write_value(out, item, ascii, compact);
            }
            out.push('}');
        }
    }
}

/// The wire form of every response (ASCII-escaped, like Python's default `json.dumps`).
pub fn to_string(v: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, v, true, false);
    out
}

/// Sorted keys, compact separators, UTF-8 kept: the form registration digests hash.
pub fn canonical(v: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, v, false, true);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_types() {
        let v = parse(br#"{"b": [1, 2.5, -3e-5, true, null], "a": "x\u00e9\n"}"#).unwrap();
        assert_eq!(canonical(&v), "{\"a\":\"xé\\n\",\"b\":[1,2.5,-3e-05,true,null]}");
        assert!(matches!(v.get("b").unwrap().as_array().unwrap()[0], Value::Int(1)));
        assert!(matches!(parse(b"64.0").unwrap(), Value::Float(_)));
        assert!(parse(b"{\"a\":1,}").is_err());
        assert!(parse(b"01").is_err());
        assert_eq!(float_repr(6.0), "6.0");
        assert_eq!(float_repr(0.8), "0.8");
        assert_eq!(float_repr(1e16), "1e+16");
        assert_eq!(float_repr(0.0001), "0.0001");
        assert_eq!(float_repr(0.00001), "1e-05");
        assert_eq!(to_string(&Value::str("é")), "\"\\u00e9\"");
        assert_eq!(Value::Int(1), Value::Float(1.0));
    }
}
