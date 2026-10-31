//! Owned process plumbing: random identifiers, UTC timestamps, signals, memory facts.

use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// 32 lowercase hex characters from the kernel's CSPRNG (a UUID4's worth of randomness).
pub fn random_hex() -> String {
    let mut bytes = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut bytes))
        .expect("/dev/urandom is readable");
    crate::sha256::hex(&bytes)
}

/// A random UUID4 in its canonical text form (session IDs).
pub fn uuid4() -> String {
    let mut bytes = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut bytes))
        .expect("/dev/urandom is readable");
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let h = crate::sha256::hex(&bytes);
    format!("{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32])
}

/// `datetime.now(timezone.utc).isoformat()`: `2026-10-31T13:41:00.123456+00:00`.
pub fn utc_now() -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = now.as_secs() as i64;
    let micros = now.subsec_micros();
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    // Civil from days (Howard Hinnant's algorithm).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}+00:00",
        y,
        m,
        d,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60,
        micros
    )
}

// ----- signals ---------------------------------------------------------------------------------

pub static STOP: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn signal(signum: i32, handler: usize) -> usize;
}

extern "C" fn on_signal(_signum: i32) {
    STOP.store(true, Ordering::SeqCst);
}

/// SIGTERM and SIGINT request a cooperative stop (admission ends, the running execution is
/// cancelled), as the Python worker's handler does.
pub fn install_stop_handlers() {
    const SIGINT: i32 = 2;
    const SIGTERM: i32 = 15;
    let handler = on_signal as extern "C" fn(i32) as usize;
    unsafe {
        signal(SIGTERM, handler);
        signal(SIGINT, handler);
    }
}

// ----- memory ----------------------------------------------------------------------------------

fn status_kib(field: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    text.lines()
        .find(|l| l.starts_with(field))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
}

/// The process's peak resident set in MiB (`VmHWM`), like `ru_maxrss / 1024` in Python.
pub fn peak_rss_mb() -> Option<f64> {
    status_kib("VmHWM:").map(|kib| kib as f64 / 1024.0)
}

pub fn total_memory_mb() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    text.lines()
        .find(|l| l.starts_with("MemTotal:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<u64>().ok())
        .map(|kib| kib / 1024)
}

/// A stable host identifier that reveals neither the machine ID nor the hostname.
pub fn host_identity() -> (String, &'static str) {
    if let Ok(machine) = std::fs::read_to_string("/etc/machine-id") {
        let machine = machine.trim();
        if !machine.is_empty() {
            return (crate::sha256::hex_digest(machine.as_bytes()), "machine-id");
        }
    }
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default();
    (crate::sha256::hex_digest(host.trim().as_bytes()), "hostname")
}
