//! Typed reads of a component's JSON config; anything outside the supported subset is
//! `Error::Unsupported("<component>.<field>")`.

use std::path::Path;

use crate::json::{self, Value};
use crate::Error;

pub struct Config {
    pub component: String,
    pub value: Value,
}

type Result<T> = std::result::Result<T, Error>;

impl Config {
    pub fn read(path: &Path, component: &str) -> Result<Config> {
        let text = std::fs::read_to_string(path).map_err(|e| Error::Failed(format!("{component} config: {e}")))?;
        let value = json::parse(&text).map_err(|e| Error::Failed(format!("{component} config: {e}")))?;
        if value.as_object().is_none() {
            return Err(Error::Failed(format!("{component} config is not an object")));
        }
        Ok(Config { component: component.to_string(), value })
    }

    pub fn unsupported(&self, field: &str) -> Error {
        Error::Unsupported(format!("{}.{}", self.component, field))
    }

    /// The field's value, `None` when absent or null.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.value.get(key).filter(|v| !v.is_null())
    }

    pub fn usize_or(&self, key: &str, default: usize) -> Result<usize> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v.as_i64().filter(|i| *i >= 0).map(|i| i as usize).ok_or_else(|| self.unsupported(key)),
        }
    }

    pub fn opt_usize(&self, key: &str) -> Result<Option<usize>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v.as_i64().filter(|i| *i >= 0).map(|i| Some(i as usize)).ok_or_else(|| self.unsupported(key)),
        }
    }

    pub fn i64_or(&self, key: &str, default: i64) -> Result<i64> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v.as_i64().ok_or_else(|| self.unsupported(key)),
        }
    }

    pub fn f64_or(&self, key: &str, default: f64) -> Result<f64> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v.as_f64().filter(|f| f.is_finite()).ok_or_else(|| self.unsupported(key)),
        }
    }

    /// A boolean flag; an explicit null is false (the model classes test these with `if flag`).
    pub fn bool_or(&self, key: &str, default: bool) -> Result<bool> {
        match self.value.get(key) {
            None => Ok(default),
            Some(Value::Null) => Ok(false),
            Some(v) => v.as_bool().ok_or_else(|| self.unsupported(key)),
        }
    }

    pub fn opt_str(&self, key: &str) -> Result<Option<String>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v.as_str().map(|s| Some(s.to_string())).ok_or_else(|| self.unsupported(key)),
        }
    }

    pub fn str_or(&self, key: &str, default: &str) -> Result<String> {
        Ok(self.opt_str(key)?.unwrap_or_else(|| default.to_string()))
    }

    /// The field must be absent/null, or (when given) equal to the default.
    pub fn require_null(&self, key: &str) -> Result<()> {
        if self.get(key).is_some() {
            return Err(self.unsupported(key));
        }
        Ok(())
    }

    pub fn require_bool(&self, key: &str, allowed: bool) -> Result<()> {
        if self.bool_or(key, allowed)? != allowed {
            return Err(self.unsupported(key));
        }
        Ok(())
    }

    pub fn require_str(&self, key: &str, default: &str, allowed: &[&str]) -> Result<String> {
        let value = self.str_or(key, default)?;
        if !allowed.contains(&value.as_str()) {
            return Err(self.unsupported(key));
        }
        Ok(value)
    }

    pub fn str_list(&self, key: &str, default: &[&str]) -> Result<Vec<String>> {
        match self.get(key) {
            None => Ok(default.iter().map(|s| s.to_string()).collect()),
            Some(v) => v
                .as_array()
                .ok_or_else(|| self.unsupported(key))?
                .iter()
                .map(|x| x.as_str().map(str::to_string).ok_or_else(|| self.unsupported(key)))
                .collect(),
        }
    }

    pub fn usize_list(&self, key: &str, default: &[usize]) -> Result<Vec<usize>> {
        match self.get(key) {
            None => Ok(default.to_vec()),
            Some(v) => v
                .as_array()
                .ok_or_else(|| self.unsupported(key))?
                .iter()
                .map(|x| x.as_i64().filter(|i| *i > 0).map(|i| i as usize).ok_or_else(|| self.unsupported(key)))
                .collect(),
        }
    }

    /// An int broadcast to `n` entries, or a list of exactly `n` ints.
    pub fn per_block(&self, key: &str, default: usize, n: usize) -> Result<Vec<usize>> {
        match self.get(key) {
            None => Ok(vec![default; n]),
            Some(Value::Array(items)) => {
                if items.len() != n {
                    return Err(self.unsupported(key));
                }
                items
                    .iter()
                    .map(|x| x.as_i64().filter(|i| *i > 0).map(|i| i as usize).ok_or_else(|| self.unsupported(key)))
                    .collect()
            }
            Some(v) => v.as_i64().filter(|i| *i > 0).map(|i| vec![i as usize; n]).ok_or_else(|| self.unsupported(key)),
        }
    }
}
