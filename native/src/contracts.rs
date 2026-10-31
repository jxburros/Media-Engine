//! Validation shared by every endpoint, mirroring `src/media_engine/contracts.py` for the
//! families and tasks the native engine serves (sdxl-base: text-to-image, image-to-image).
//! Never accepts executable model code. Error codes and fields are the Python worker's.

use crate::json::{canonical, Map, Value};
use crate::obj;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const PROTOCOL: &str = "media-worker/1.3";
pub const SUPPORTED_PROTOCOLS: [&str; 4] = ["media-worker/1.0", "media-worker/1.1", "media-worker/1.2", "media-worker/1.3"];
pub const WORKER_VERSION: &str = env!("CARGO_PKG_VERSION");
/// Every task name the protocol knows, and the minor that introduced it.
pub const ALL_TASKS: [(&str, u32); 5] =
    [("text-to-image", 0), ("image-to-image", 1), ("upscale", 2), ("image-to-video", 3), ("text-to-audio", 3)];
/// The tasks this engine serves.
pub const TASKS: [&str; 2] = ["text-to-image", "image-to-image"];
/// Every family name the protocol knows, and the minor that introduced it.
pub const ALL_FAMILIES: [(&str, u32); 4] = [("sdxl-base", 0), ("esrgan", 2), ("svd", 3), ("stable-audio-open", 3)];
pub const FAMILY: &str = "sdxl-base";
pub const ROLES: [&str; 7] = ["unet", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"];
pub const WEIGHT_ROLES: [&str; 4] = ["unet", "vae", "text_encoder", "text_encoder_2"];
pub const APPROVED: [(&str, &str, &str); 7] = [
    ("unet", "diffusers", "UNet2DConditionModel"),
    ("vae", "diffusers", "AutoencoderKL"),
    ("text_encoder", "transformers", "CLIPTextModel"),
    ("text_encoder_2", "transformers", "CLIPTextModelWithProjection"),
    ("tokenizer", "transformers", "CLIPTokenizer"),
    ("tokenizer_2", "transformers", "CLIPTokenizer"),
    ("scheduler", "diffusers", "EulerDiscreteScheduler"),
];
pub const FORMATS: [&str; 1] = ["diffusers-safetensors"];
pub const MAX_ADAPTERS: usize = 4;
pub const ADAPTER_TARGETS: [&str; 3] = ["unet", "text_encoder", "text_encoder_2"];
pub const ADAPTER_FORMAT: &str = "peft-lora";
pub const AUTO_OPTIONS: [&str; 1] = ["precision"];
pub const FALLBACKS: [&str; 2] = ["cpu", "lower_precision"];
const FORBIDDEN_SUFFIXES: [&str; 6] = ["py", "pkl", "pickle", "bin", "pt", "pth"];

#[derive(Debug, Clone)]
pub struct WorkerError {
    pub status: u16,
    pub code: &'static str,
    pub field: Option<String>,
}

pub fn fail(status: u16, code: &'static str, field: Option<&str>) -> WorkerError {
    WorkerError { status, code, field: field.map(str::to_string) }
}
pub fn bad(code: &'static str, field: &str) -> WorkerError {
    fail(400, code, Some(field))
}
pub fn invalid() -> WorkerError {
    fail(400, "invalid_request", None)
}

pub type Result<T> = std::result::Result<T, WorkerError>;

pub fn limits() -> Value {
    obj! {
        "width" => vec![Value::Int(64), Value::Int(2048)],
        "height" => vec![Value::Int(64), Value::Int(2048)],
        "steps" => vec![Value::Int(1), Value::Int(150)],
        "guidance" => vec![Value::Int(0), Value::Int(30)],
        "strength" => vec![Value::Int(0), Value::Int(1)],
        "adapter_scale" => vec![Value::Int(0), Value::Int(2)],
        "size_multiple" => 8i64,
        "count" => vec![Value::Int(1), Value::Int(4)],
    }
}

pub fn defaults() -> Value {
    obj! {"width" => 1024i64, "height" => 1024i64, "steps" => 30i64, "guidance" => 6i64, "sampler" => "euler", "scheduler" => "euler"}
}

pub fn task_defaults() -> Value {
    obj! {"image-to-image" => obj!{"strength" => 0.75}}
}

/// `strict()`: an object with every required key and nothing beyond required ∪ optional.
pub fn strict<'a>(value: &'a Value, required: &[&str], optional: &[&str]) -> Result<&'a Map> {
    let map = value.as_object().ok_or_else(invalid)?;
    if required.iter().any(|k| !map.contains_key(*k))
        || map.keys().any(|k| !required.contains(&k.as_str()) && !optional.contains(&k.as_str()))
    {
        return Err(invalid());
    }
    Ok(map)
}

pub fn minor(version: &str) -> u32 {
    version.rsplit('.').next().and_then(|m| m.parse().ok()).unwrap_or(0)
}

/// The request's protocol string, which must be one this engine serves.
pub fn protocol(body: &Value, served: &[String]) -> Result<String> {
    match body.get("protocol").and_then(Value::as_str) {
        Some(v) if SUPPORTED_PROTOCOLS.contains(&v) => {
            if !served.iter().any(|s| s == v) {
                return Err(bad("protocol_mismatch", "protocol"));
            }
            Ok(v.to_string())
        }
        _ => Err(bad("protocol_mismatch", "protocol")),
    }
}

pub fn digest(value: &Value) -> String {
    format!("sha256:{}", crate::sha256::hex_digest(canonical(value).as_bytes()))
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub fn is_sha256_ref(s: &str) -> bool {
    s.strip_prefix("sha256:").is_some_and(is_hex64)
}
fn is_ident(s: &str, max: usize) -> bool {
    (1..=max).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}
pub fn is_input_id(s: &str) -> bool {
    s.strip_prefix("input_").is_some_and(|h| h.len() == 32 && h.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
}
pub fn is_handle(prefix: &str, s: &str) -> bool {
    s.strip_prefix(prefix).is_some_and(|h| h.len() == 32 && h.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
}

fn role_files(role: &str, files: &Map) -> Map {
    files.iter().filter(|(k, _)| k.starts_with(&format!("{}/", role))).map(|(k, v)| (k.clone(), v.clone())).collect()
}

fn walk(root: &Path, base: &Path, out: &mut BTreeMap<String, PathBuf>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(root).map_err(|_| bad("missing_component", "configuration"))?.flatten().collect();
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let path = entry.path();
        let meta = std::fs::symlink_metadata(&path).map_err(|_| bad("missing_component", "configuration"))?;
        if meta.file_type().is_symlink() {
            return Err(bad("unsupported", "symlink"));
        }
        if meta.is_dir() {
            walk(&path, base, out)?;
        } else if meta.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if FORBIDDEN_SUFFIXES.contains(&ext) {
                    return Err(bad("unsupported", "tensor_format"));
                }
            }
            let relative = path.strip_prefix(base).unwrap().to_string_lossy().to_string();
            out.insert(relative, path);
        }
    }
    Ok(())
}

fn read_json(path: &Path) -> Option<Value> {
    crate::json::parse(&std::fs::read(path).ok()?).ok()
}

fn check_conversion(conversion: &Value) -> Result<()> {
    let invalid = || bad("invalid_request", "conversion");
    let map = strict(
        conversion,
        &["source_sha256", "reference_bundle_digest", "converter", "mapped_keys"],
        &["schema_version", "precision", "unmapped_keys", "ignored_keys"],
    )?;
    if !map["source_sha256"].as_str().is_some_and(is_hex64) {
        return Err(invalid());
    }
    if !map["reference_bundle_digest"].as_str().is_some_and(is_sha256_ref) {
        return Err(invalid());
    }
    let converter = map["converter"].as_object().ok_or_else(invalid)?;
    let names: Vec<&str> = converter.keys().map(String::as_str).collect();
    if names != ["diffusers", "media_engine", "safetensors", "torch"]
        || converter.values().any(|v| !matches!(v, Value::Null | Value::Str(_)))
    {
        return Err(invalid());
    }
    let mapped = map["mapped_keys"].as_object().ok_or_else(invalid)?;
    let mut keys: Vec<&str> = mapped.keys().map(String::as_str).collect();
    keys.sort();
    let mut expected = WEIGHT_ROLES.to_vec();
    expected.sort();
    if keys != expected || mapped.values().any(|v| !v.as_int().is_some_and(|i| i >= 0)) {
        return Err(invalid());
    }
    if map.get("schema_version").is_some_and(|v| *v != Value::Int(1))
        || map.get("unmapped_keys").is_some_and(|v| *v != Value::Array(vec![]))
    {
        return Err(invalid());
    }
    if let Some(p) = map.get("precision") {
        if !matches!(p.as_str(), Some("preserved" | "float16" | "float32")) {
            return Err(invalid());
        }
    }
    if let Some(ignored) = map.get("ignored_keys") {
        if !ignored.as_array().is_some_and(|a| a.iter().all(|k| k.as_str().is_some())) {
            return Err(invalid());
        }
    }
    Ok(())
}

/// `bundle_path()` for sdxl-base: check the registration, verify every file, return the root.
/// Callers have already refused families this engine does not implement.
pub fn bundle_path(bundle: &Value, catalog: &BTreeMap<String, String>, verify: bool, version: &str) -> Result<PathBuf> {
    let level = minor(version);
    let map = strict(
        bundle,
        &["schema_version", "bundle_id", "bundle_digest", "family", "pipeline", "components", "configuration", "source"],
        if level >= 2 { &["format"] } else { &[] },
    )?;
    let family = map["family"].as_str();
    let since = ALL_FAMILIES.iter().find(|(n, _)| Some(*n) == family).map(|(_, s)| *s);
    if map["schema_version"] != Value::Int(1)
        || family != Some(FAMILY)
        || since.is_none_or(|s| s > level)
        || map.get("format").is_some_and(|f| f.as_str() != Some("diffusers-safetensors"))
        || map["pipeline"].as_str() != Some("text-to-image")
    {
        return Err(bad("unsupported", "family"));
    }
    let mut unsigned = map.clone();
    unsigned.remove("bundle_digest");
    if map["bundle_digest"].as_str() != Some(digest(&Value::Object(unsigned)).as_str()) {
        return Err(bad("artifact_changed", "bundle_digest"));
    }
    if !map["bundle_id"].as_str().is_some_and(|s| is_ident(s, 80)) {
        return Err(bad("invalid_request", "bundle_id"));
    }
    let source_optional: &[&str] =
        if level >= 2 { &["repository", "revision", "license", "conversion"] } else { &["repository", "revision", "license"] };
    let source = strict(&map["source"], &[], source_optional)?;
    if source.iter().any(|(k, v)| k != "conversion" && v.as_str().is_none()) {
        return Err(bad("invalid_request", "source"));
    }
    let conversion = source.get("conversion");
    if let Some(c) = conversion {
        check_conversion(c)?;
    }
    let configuration = strict(&map["configuration"], &["artifact_handle", "files"], &[])?;
    let handle = configuration["artifact_handle"].as_str().filter(|h| catalog.contains_key(*h));
    let Some(handle) = handle else { return Err(bad("missing_component", "artifact_handle")) };
    let root = PathBuf::from(&catalog[handle]);
    let root = root.canonicalize().unwrap_or(root);
    let files = match configuration["files"].as_object() {
        Some(f) if !f.is_empty() => f,
        _ => return Err(bad("missing_component", "configuration")),
    };
    let components = match map["components"].as_array() {
        Some(c) if c.len() == ROLES.len() => c,
        _ => return Err(bad("missing_component", "components")),
    };
    let mut seen = Vec::new();
    for component in components {
        let c = strict(component, &["role", "artifact_handle", "digest", "tensor_format"], &[])?;
        let role = c["role"].as_str().unwrap_or("");
        if !ROLES.contains(&role) || seen.contains(&role) || c["artifact_handle"].as_str() != Some(handle) {
            return Err(bad("unsupported", "components"));
        }
        seen.push(role);
        let entries = role_files(role, files);
        if entries.is_empty() || c["digest"].as_str() != Some(digest(&Value::Object(entries.clone())).as_str()) {
            return Err(bad("artifact_changed", role));
        }
        let expected = if WEIGHT_ROLES.contains(&role) { "safetensors" } else { "json" };
        if c["tensor_format"].as_str() != Some(expected) {
            return Err(bad("unsupported", role));
        }
        if expected == "safetensors" && !entries.keys().any(|k| k.ends_with(".safetensors")) {
            return Err(bad("missing_component", role));
        }
    }
    if !root.is_dir() {
        return Err(bad("missing_component", "configuration"));
    }
    let mut actual = BTreeMap::new();
    walk(&root, &root, &mut actual)?;
    if actual.keys().ne(files.keys()) {
        return Err(bad("artifact_changed", "configuration"));
    }
    for (relative, expected) in files {
        let Some(expected) = expected.as_str().filter(|e| is_hex64(e)) else {
            return Err(bad("invalid_request", "digest"));
        };
        if verify {
            let actual_digest = crate::sha256::file_digest(&actual[relative]).map_err(|_| bad("missing_component", "configuration"))?;
            if actual_digest != expected {
                return Err(bad("artifact_changed", relative.split('/').next().unwrap()));
            }
        }
    }
    if let Some(conversion) = conversion {
        let recorded = actual.get("conversion.json").and_then(|p| read_json(p));
        let matches = recorded.as_ref().and_then(Value::as_object).is_some_and(|r| {
            conversion.as_object().unwrap().iter().all(|(k, v)| r.get(k) == Some(v))
        });
        if !matches {
            return Err(bad("artifact_changed", "conversion"));
        }
    }
    let index = read_json(&root.join("model_index.json"));
    let scheduler = read_json(&root.join("scheduler/scheduler_config.json"));
    let (Some(index), Some(scheduler)) = (index, scheduler) else {
        return Err(bad("missing_component", "configuration"));
    };
    let index = index.as_object().ok_or_else(|| bad("missing_component", "configuration"))?;
    if index.get("_class_name").and_then(Value::as_str) != Some("StableDiffusionXLPipeline")
        || scheduler.get("_class_name").and_then(Value::as_str) != Some("EulerDiscreteScheduler")
    {
        return Err(bad("unsupported", "pipeline"));
    }
    for (key, value) in index {
        if key.starts_with('_') || APPROVED.iter().any(|(r, _, _)| r == key) {
            continue;
        }
        if key == "force_zeros_for_empty_prompt" && matches!(value, Value::Bool(_)) {
            continue;
        }
        if (key == "feature_extractor" || key == "image_encoder") && *value == Value::Array(vec![Value::Null, Value::Null]) {
            continue;
        }
        return Err(bad("unsupported", "pipeline"));
    }
    for (role, library, class) in APPROVED {
        if index.get(role) != Some(&Value::Array(vec![Value::str(library), Value::str(class)])) {
            return Err(bad("unsupported", role));
        }
    }
    Ok(root)
}

// ----- safetensors headers and the memory estimate -------------------------------------------

pub fn safetensors_header(path: &Path) -> Result<Map> {
    use std::io::Read;
    let unsupported = || bad("unsupported", "tensor_format");
    let mut file = std::fs::File::open(path).map_err(|_| unsupported())?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length).map_err(|_| unsupported())?;
    let length = u64::from_le_bytes(length);
    if !(1..=64 * 1024 * 1024).contains(&length) {
        return Err(unsupported());
    }
    let mut header = vec![0u8; length as usize];
    file.read_exact(&mut header).map_err(|_| unsupported())?;
    match crate::json::parse(&header) {
        Ok(Value::Object(map)) => Ok(map),
        _ => Err(unsupported()),
    }
}

fn dtype_bytes(dtype: &str) -> u64 {
    match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" => 1,
        _ => 4,
    }
}

fn shape_count(tensor: &Value) -> u64 {
    tensor.get("shape").and_then(Value::as_array).map_or(0, |s| s.iter().map(|d| d.as_int().unwrap_or(0) as u64).product())
}

pub fn precision_bytes(precision: &str) -> u64 {
    if precision == "float32" { 4 } else { 2 }
}

/// Parameter bytes per role at the load precision (the Python worker's sdxl-base estimate).
pub fn memory_estimate(root: &Path, precision: &str, device: &str) -> Result<Value> {
    let per = precision_bytes(precision);
    let mut roles = Map::new();
    let mut total = 0u64;
    for role in WEIGHT_ROLES {
        let mut files: Vec<PathBuf> = std::fs::read_dir(root.join(role))
            .map(|d| d.flatten().map(|e| e.path()).filter(|p| p.to_string_lossy().ends_with(".safetensors")).collect())
            .unwrap_or_default();
        files.sort();
        let (mut parameters, mut stored) = (0u64, 0u64);
        for path in files {
            for (name, tensor) in safetensors_header(&path)? {
                if name == "__metadata__" {
                    continue;
                }
                let count = shape_count(&tensor);
                parameters += count;
                stored += count * dtype_bytes(tensor.get("dtype").and_then(Value::as_str).unwrap_or(""));
            }
        }
        let loaded = parameters * per / (1024 * 1024);
        total += loaded;
        roles.insert(
            role.to_string(),
            obj! {"parameters" => parameters, "stored_mb" => stored / (1024 * 1024), "loaded_mb" => loaded},
        );
    }
    Ok(obj! {
        "value_mb" => total,
        "source" => "safetensors_headers",
        "basis" => "parameter count at the requested load precision",
        "precision" => precision,
        "device" => device,
        "roles" => Value::Object(roles),
        "uncertainty" => "excludes activations, VAE float32 upcast, allocator reserve and framework overhead",
    })
}

// ----- adapters ----------------------------------------------------------------------------

pub struct ResolvedAdapter {
    pub record: Value,
    pub weights: PathBuf,
    pub scale: Value,
    pub layout: Value,
}

fn adapter_path(adapter: &Value, catalog: &BTreeMap<String, String>, verify: bool) -> Result<PathBuf> {
    let map = strict(adapter, &["schema_version", "adapter_id", "artifact_handle", "files", "digest", "source"], &[])?;
    if map["schema_version"] != Value::Int(1) {
        return Err(bad("unsupported", "adapter"));
    }
    if !map["adapter_id"].as_str().is_some_and(|s| is_ident(s, 80)) {
        return Err(bad("invalid_request", "adapter_id"));
    }
    let mut unsigned = map.clone();
    unsigned.remove("digest");
    if map["digest"].as_str() != Some(digest(&Value::Object(unsigned)).as_str()) {
        return Err(bad("artifact_changed", "adapter"));
    }
    let source = strict(&map["source"], &[], &["repository", "revision", "license"])?;
    if source.values().any(|v| v.as_str().is_none()) {
        return Err(bad("invalid_request", "source"));
    }
    let Some(handle) = map["artifact_handle"].as_str().filter(|h| catalog.contains_key(*h)) else {
        return Err(bad("missing_component", "adapter"));
    };
    let root = PathBuf::from(&catalog[handle]);
    let root = root.canonicalize().unwrap_or(root);
    let files = match map["files"].as_object() {
        Some(f) if !f.is_empty() && root.is_dir() => f,
        _ => return Err(bad("missing_component", "adapter")),
    };
    let mut actual = BTreeMap::new();
    walk(&root, &root, &mut actual)?;
    if actual.keys().ne(files.keys()) {
        return Err(bad("artifact_changed", "adapter"));
    }
    let weights: Vec<&String> = files.keys().filter(|k| k.ends_with(".safetensors")).collect();
    if weights.len() != 1 || weights[0].contains('/') {
        return Err(bad("unsupported", "adapter"));
    }
    for (relative, expected) in files {
        let Some(expected) = expected.as_str().filter(|e| is_hex64(e)) else {
            return Err(bad("invalid_request", "digest"));
        };
        if verify && crate::sha256::file_digest(&actual[relative]).ok().as_deref() != Some(expected) {
            return Err(bad("artifact_changed", "adapter"));
        }
    }
    Ok(actual[weights[0]].clone())
}

fn lora_layout(path: &Path) -> Result<Value> {
    let unsupported = || bad("unsupported", "adapter");
    let header = safetensors_header(path).map_err(|_| unsupported())?;
    let mut modules: BTreeMap<(String, String), BTreeMap<String, Vec<i64>>> = BTreeMap::new();
    let mut parameters = 0u64;
    for (name, tensor) in &header {
        if name == "__metadata__" {
            continue;
        }
        let (target, rest) = name.split_once('.').ok_or_else(unsupported)?;
        if !ADAPTER_TARGETS.contains(&target) {
            return Err(unsupported());
        }
        let (module, part) = if let Some(m) = rest.strip_suffix(".lora_A.weight") {
            (m, "lora_A.weight")
        } else if let Some(m) = rest.strip_suffix(".lora_B.weight") {
            (m, "lora_B.weight")
        } else if let Some(m) = rest.strip_suffix(".alpha") {
            (m, "alpha")
        } else {
            return Err(unsupported());
        };
        if module.is_empty() {
            return Err(unsupported());
        }
        let shape: Vec<i64> = tensor.get("shape").and_then(Value::as_array).map_or(vec![], |s| s.iter().filter_map(Value::as_int).collect());
        parameters += shape_count(tensor);
        modules.entry((target.to_string(), module.to_string())).or_default().insert(part.to_string(), shape);
    }
    if modules.is_empty() {
        return Err(unsupported());
    }
    let mut ranks = std::collections::BTreeSet::new();
    for parts in modules.values() {
        let (Some(a), Some(b)) = (parts.get("lora_A.weight"), parts.get("lora_B.weight")) else { return Err(unsupported()) };
        if a.first() != b.get(1) {
            return Err(unsupported());
        }
        ranks.insert(a[0]);
    }
    let targets: std::collections::BTreeSet<&String> = modules.keys().map(|(t, _)| t).collect();
    Ok(obj! {
        "format" => ADAPTER_FORMAT,
        "targets" => Value::Array(targets.into_iter().map(|t| Value::str(t.as_str())).collect()),
        "modules" => modules.len(),
        "ranks" => Value::Array(ranks.into_iter().map(Value::Int).collect()),
        "parameters" => parameters,
    })
}

pub fn validate_adapters(adapters: &Value, catalog: &BTreeMap<String, String>, version: &str, verify: bool) -> Result<Vec<ResolvedAdapter>> {
    let Some(list) = adapters.as_array().filter(|a| a.len() <= MAX_ADAPTERS) else {
        return Err(bad("invalid_request", "adapters"));
    };
    if !list.is_empty() && version == "media-worker/1.0" {
        return Err(bad("invalid_request", "adapters"));
    }
    let mut resolved: Vec<ResolvedAdapter> = Vec::new();
    for entry in list {
        let map = strict(entry, &["adapter", "scale"], &[])?;
        let scale = &map["scale"];
        if !scale.as_number().is_some_and(|s| (0.0..=2.0).contains(&s)) {
            return Err(bad("invalid_request", "scale"));
        }
        let weights = adapter_path(&map["adapter"], catalog, verify)?;
        let id = map["adapter"].get("adapter_id").cloned();
        if resolved.iter().any(|r| r.record.get("adapter_id").cloned() == id) {
            return Err(bad("invalid_request", "adapter_id"));
        }
        let layout = lora_layout(&weights)?;
        resolved.push(ResolvedAdapter { record: map["adapter"].clone(), weights, scale: scale.clone(), layout });
    }
    Ok(resolved)
}

// ----- executions ----------------------------------------------------------------------------

pub fn effective_steps(parameters: &Map, task: &str) -> i64 {
    let steps = parameters.get("steps").and_then(Value::as_int).unwrap_or(0);
    if task != "image-to-image" {
        return steps;
    }
    let strength = parameters.get("strength").and_then(Value::as_number).unwrap_or(0.0);
    ((steps as f64 * strength) as i64).min(steps)
}

pub fn output_ids(task: &str, parameters: &Map) -> Vec<String> {
    let count = if task == "text-to-image" { parameters.get("count").and_then(Value::as_int).unwrap_or(1) } else { 1 };
    (1..=count).map(|k| if k == 1 { "image".to_string() } else { format!("image_{}", k) }).collect()
}

pub fn effective_settings(parameters: &Map, task: &str, version: &str) -> Value {
    let mut settings = parameters.clone();
    settings.insert("effective_steps".into(), Value::Int(effective_steps(parameters, task)));
    if task == "text-to-image" && minor(version) >= 2 && !settings.contains_key("count") {
        settings.insert("count".into(), Value::Int(1));
    }
    Value::Object(settings)
}

/// `validate_parameters()`: the execution body's shape and ranges; returns the task.
pub fn validate_parameters(body: &Value, served: &[String]) -> Result<String> {
    let map = strict(body, &["protocol", "attempt_id", "bundle_digest", "task", "inputs", "parameters"], &[])?;
    let version = protocol(body, served)?;
    let level = minor(&version);
    if !map["bundle_digest"].as_str().is_some_and(is_sha256_ref) {
        return Err(bad("invalid_request", "bundle_digest"));
    }
    let task = map["task"].as_str().unwrap_or("");
    let known = ALL_TASKS.iter().find(|(t, _)| *t == task);
    if known.is_none_or(|(_, since)| *since > level) || !TASKS.contains(&task) {
        // A task newer than the caller's minor, or one this engine does not serve.
        return Err(bad("unsupported", "task"));
    }
    let attempt_ok = map["attempt_id"].as_str().is_some_and(|a| {
        a.strip_prefix("attempt_").is_some_and(|rest| is_ident(rest, 80))
    });
    if !attempt_ok {
        return Err(bad("invalid_request", "attempt_id"));
    }
    let required_inputs: &[&str] = if task == "image-to-image" { &["prompt", "image"] } else { &["prompt"] };
    let inputs = strict(&map["inputs"], required_inputs, &["negative_prompt"])?;
    for (key, value) in inputs {
        if key == "image" {
            if !value.as_str().is_some_and(is_input_id) {
                return Err(bad("invalid_request", key));
            }
            continue;
        }
        let ok = value.as_str().is_some_and(|s| s.chars().count() <= 10000 && (key != "prompt" || !s.trim().is_empty()));
        if !ok {
            return Err(bad("invalid_request", key));
        }
    }
    if let Some(params) = map["parameters"].as_object() {
        if let Some(count) = params.get("count") {
            if level < 2 || task != "text-to-image" {
                return Err(bad("unsupported", "count"));
            }
            if !count.as_int().is_some_and(|c| (1..=4).contains(&c)) {
                return Err(bad("invalid_request", "count"));
            }
        }
    }
    let mut required = vec!["width", "height", "steps", "guidance", "seed", "sampler", "scheduler"];
    if task == "image-to-image" {
        required.push("strength");
    }
    let optional: &[&str] = if task == "text-to-image" && level >= 2 { &["count"] } else { &[] };
    let params = strict(&map["parameters"], &required, optional)?;
    for (field, low, high, multiple) in [("width", 64, 2048, 8), ("height", 64, 2048, 8), ("steps", 1, 150, 1), ("seed", 0, 4294967295i64, 1)] {
        let ok = params[field].as_int().is_some_and(|v| (low..=high).contains(&v) && v % multiple == 0);
        if !ok {
            return Err(bad("invalid_request", field));
        }
    }
    let count = params.get("count").and_then(Value::as_int).unwrap_or(1);
    if count > 1 && params["seed"].as_int().unwrap() + count - 1 > 4294967295 {
        return Err(bad("invalid_request", "seed"));
    }
    if !params["guidance"].as_number().is_some_and(|g| (0.0..=30.0).contains(&g)) {
        return Err(bad("invalid_request", "guidance"));
    }
    if task == "image-to-image" {
        if !params["strength"].as_number().is_some_and(|s| s > 0.0 && s <= 1.0) {
            return Err(bad("invalid_request", "strength"));
        }
        if effective_steps(params, task) < 1 {
            return Err(bad("invalid_request", "strength"));
        }
    }
    if params["sampler"].as_str() != Some("euler") || params["scheduler"].as_str() != Some("euler") {
        return Err(bad("unsupported", "scheduler"));
    }
    Ok(task.to_string())
}

/// Validate a PNG signature/IHDR and return (width, height) without decoding pixels.
pub fn png_dimensions(data: &[u8]) -> Result<(u32, u32)> {
    if data.len() < 33 || &data[..8] != b"\x89PNG\r\n\x1a\n" || &data[12..16] != b"IHDR" {
        return Err(bad("invalid_request", "image"));
    }
    let width = u32::from_be_bytes(data[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(data[20..24].try_into().unwrap());
    let (depth, color) = (data[24], data[25]);
    if depth != 8 || ![0u8, 2, 4, 6].contains(&color) || &data[data.len() - 8..data.len() - 4] != b"IEND" {
        return Err(bad("unsupported", "image"));
    }
    for value in [width, height] {
        if !(64..=2048).contains(&value) || value % 8 != 0 {
            return Err(bad("invalid_request", "image"));
        }
    }
    Ok((width, height))
}
