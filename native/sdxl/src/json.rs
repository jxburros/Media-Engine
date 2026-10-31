//! A small owned JSON reader for the bundle's config, scheduler, tokenizer and index files.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// The number as written (kept so integers stay exact) and its f64 value.
    Number(String, f64),
    String(String),
    Array(Vec<Value>),
    /// Members in document order; a repeated key resolves to its last occurrence, as Python's `json`.
    Object(Vec<(String, Value)>),
}

#[derive(Debug)]
pub struct ParseError {
    pub offset: usize,
    pub message: &'static str,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid JSON at byte {}: {}", self.offset, self.message)
    }
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(members) => members.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(_, v) => Some(*v),
            _ => None,
        }
    }

    /// An integral number (written without fraction/exponent, or an exactly integral float).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Number(raw, v) => {
                if let Ok(i) = raw.parse::<i64>() {
                    Some(i)
                } else if v.fract() == 0.0 && v.abs() < 9.0e15 {
                    Some(*v as i64)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Object(members) => Some(members),
            _ => None,
        }
    }
}

pub fn parse(text: &str) -> Result<Value, ParseError> {
    let mut parser = Parser { bytes: text.as_bytes(), pos: 0 };
    // A UTF-8 byte-order mark is tolerated, as Python's json does not but its open(encoding=utf-8-sig) callers do.
    if parser.bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        parser.pos = 3;
    }
    parser.skip_ws();
    let value = parser.value(0)?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(parser.error("trailing characters"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn error(&self, message: &'static str) -> ParseError {
        ParseError { offset: self.pos, message }
    }

    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn expect_literal(&mut self, literal: &str, value: Value) -> Result<Value, ParseError> {
        if self.bytes[self.pos..].starts_with(literal.as_bytes()) {
            self.pos += literal.len();
            Ok(value)
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, ParseError> {
        if depth > 256 {
            return Err(self.error("nesting too deep"));
        }
        match self.peek() {
            None => Err(self.error("unexpected end")),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b't') => self.expect_literal("true", Value::Bool(true)),
            Some(b'f') => self.expect_literal("false", Value::Bool(false)),
            Some(b'n') => self.expect_literal("null", Value::Null),
            // Python's json accepts these non-standard constants; diffusers configs never use them,
            // but reading them keeps a hand-edited file from failing obscurely.
            Some(b'N') => self.expect_literal("NaN", Value::Number("NaN".into(), f64::NAN)),
            Some(b'I') => self.expect_literal("Infinity", Value::Number("Infinity".into(), f64::INFINITY)),
            Some(b'-') if self.bytes[self.pos..].starts_with(b"-Infinity") => {
                self.pos += 9;
                Ok(Value::Number("-Infinity".into(), f64::NEG_INFINITY))
            }
            Some(b'-') | Some(b'0'..=b'9') => self.number(),
            Some(_) => Err(self.error("unexpected character")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, ParseError> {
        self.pos += 1;
        let mut members = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(members));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.error("expected a key"));
            }
            let key = self.string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(self.error("expected ':'"));
            }
            self.pos += 1;
            self.skip_ws();
            let value = self.value(depth + 1)?;
            members.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Value::Object(members));
                }
                _ => return Err(self.error("expected ',' or '}'")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, ParseError> {
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.error("expected ',' or ']'")),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        let digits = self.bytes.get(self.pos..self.pos + 4).ok_or_else(|| self.error("short \\u escape"))?;
        let text = std::str::from_utf8(digits).map_err(|_| self.error("bad \\u escape"))?;
        let value = u32::from_str_radix(text, 16).map_err(|_| self.error("bad \\u escape"))?;
        self.pos += 4;
        Ok(value)
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.pos += 1;
        let mut out = String::new();
        loop {
            let start = self.pos;
            while let Some(&b) = self.bytes.get(self.pos) {
                if b == b'"' || b == b'\\' || b < 0x20 {
                    break;
                }
                self.pos += 1;
            }
            let chunk = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| self.error("invalid UTF-8"))?;
            out.push_str(chunk);
            match self.peek() {
                None => return Err(self.error("unterminated string")),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    let escape = self.peek().ok_or_else(|| self.error("unterminated escape"))?;
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let first = self.hex4()?;
                            let code =
                                if (0xD800..0xDC00).contains(&first) && self.bytes[self.pos..].starts_with(b"\\u") {
                                    let save = self.pos;
                                    self.pos += 2;
                                    let second = self.hex4()?;
                                    if (0xDC00..0xE000).contains(&second) {
                                        0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                                    } else {
                                        self.pos = save;
                                        first
                                    }
                                } else {
                                    first
                                };
                            // A lone surrogate cannot live in a Rust string; it maps to U+FFFD.
                            out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                        }
                        _ => return Err(self.error("invalid escape")),
                    }
                }
                Some(_) => return Err(self.error("control character in string")),
            }
        }
    }

    fn number(&mut self) -> Result<Value, ParseError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let digits = |p: &mut Parser| {
            let s = p.pos;
            while matches!(p.peek(), Some(b'0'..=b'9')) {
                p.pos += 1;
            }
            p.pos - s
        };
        if digits(self) == 0 {
            return Err(self.error("invalid number"));
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if digits(self) == 0 {
                return Err(self.error("invalid number"));
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if digits(self) == 0 {
                return Err(self.error("invalid number"));
            }
        }
        let raw = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap().to_string();
        let value: f64 = raw.parse().map_err(|_| self.error("invalid number"))?;
        Ok(Value::Number(raw, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_documents() {
        let v = parse(r#"{"a": [1, 2.5, -3e2, null, true], "b": {"c": "xé😀\n"}, "a": 7}"#).unwrap();
        assert_eq!(v.get("a").unwrap().as_i64(), Some(7));
        assert_eq!(v.get("b").unwrap().get("c").unwrap().as_str(), Some("x\u{e9}\u{1F600}\n"));
        assert!(parse("[1,]").is_err());
        assert!(parse("{\"a\":1} x").is_err());
    }
}
