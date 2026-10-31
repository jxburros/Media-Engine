//! Run one request through the owned native pipeline and write the PNGs.
//!
//! ```text
//! cargo run --release -p media-engine-sdxl --example oracle -- BUNDLE_DIR OUT_DIR \
//!     [--task text-to-image|image-to-image] [--prompt TEXT] [--negative TEXT] [--seed N]
//!     [--steps N] [--guidance X] [--width N] [--height N] [--count N] [--strength X]
//!     [--image INPUT.png] [--lora WEIGHTS.safetensors:SCALE]... [--precision float32|float16|bfloat16]
//!     [--device cpu|cuda:N] [--cancel-after CHECKPOINTS]
//! ```
//! Without `--negative` the request has no negative prompt. Writes `image.png`, `image_2.png`, …
//! into OUT_DIR and prints one summary line per event plus the merged-module counts and timings.

use std::path::PathBuf;
use std::time::Instant;

use media_engine_sdxl::{Adapter, DeviceSpec, Event, Pipeline, Precision, Request, Task};

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: oracle BUNDLE_DIR OUT_DIR [options]";
    let bundle = PathBuf::from(args.next().expect(usage));
    let out = PathBuf::from(args.next().expect(usage));
    let mut request = Request {
        task: Task::TextToImage,
        prompt: String::new(),
        negative_prompt: None,
        seed: 0,
        steps: 3,
        guidance: 6.0,
        width: 64,
        height: 64,
        count: 1,
        strength: 0.7,
        image_png: None,
    };
    let mut adapters = Vec::new();
    let mut precision = Precision::Float32;
    let mut device = DeviceSpec::Cpu;
    let mut cancel_after: Option<usize> = None;
    while let Some(flag) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--task" => {
                request.task = match value().as_str() {
                    "text-to-image" => Task::TextToImage,
                    "image-to-image" => Task::ImageToImage,
                    other => panic!("unknown task {other}"),
                }
            }
            "--prompt" => request.prompt = value(),
            "--negative" => request.negative_prompt = Some(value()),
            "--seed" => request.seed = value().parse().expect("seed"),
            "--steps" => request.steps = value().parse().expect("steps"),
            "--guidance" => request.guidance = value().parse().expect("guidance"),
            "--width" => request.width = value().parse().expect("width"),
            "--height" => request.height = value().parse().expect("height"),
            "--count" => request.count = value().parse().expect("count"),
            "--strength" => request.strength = value().parse().expect("strength"),
            "--image" => request.image_png = Some(std::fs::read(value()).expect("input image")),
            "--lora" => {
                let spec = value();
                let (path, scale) = spec.rsplit_once(':').expect("--lora PATH:SCALE");
                adapters.push(Adapter { weights: PathBuf::from(path), scale: scale.parse().expect("scale") });
            }
            "--precision" => {
                precision = match value().as_str() {
                    "float32" => Precision::Float32,
                    "float16" => Precision::Float16,
                    "bfloat16" => Precision::BFloat16,
                    other => panic!("unknown precision {other}"),
                }
            }
            "--device" => {
                let v = value();
                device = if v == "cpu" {
                    DeviceSpec::Cpu
                } else {
                    DeviceSpec::Cuda(v.trim_start_matches("cuda:").parse().expect("cuda index"))
                }
            }
            "--cancel-after" => cancel_after = Some(value().parse().expect("checkpoint count")),
            other => panic!("unknown flag {other}"),
        }
    }
    let started = Instant::now();
    let pipeline = match Pipeline::load(&bundle, precision, device, &adapters) {
        Ok(p) => p,
        Err(e) => {
            println!("load-error {e:?}");
            std::process::exit(2);
        }
    };
    let loaded = started.elapsed();
    println!("merged {:?}", pipeline.merged_modules());
    let started = Instant::now();
    let mut log = |event: Event| match event {
        Event::Stage(stage) => println!("stage {stage}"),
        Event::Progress { completed, total } => println!("progress {completed}/{total}"),
    };
    // `--cancel-after N`: the first N cancellation checkpoints pass, the next one cancels.
    let checks = std::cell::Cell::new(0usize);
    let cancelled = || {
        checks.set(checks.get() + 1);
        cancel_after.is_some_and(|n| checks.get() > n)
    };
    let output = match pipeline.generate(&request, &mut log, &cancelled) {
        Ok(o) => o,
        Err(media_engine_sdxl::Error::Cancelled) => {
            println!("cancelled after {} checkpoints", checks.get());
            return;
        }
        Err(e) => {
            println!("generate-error {e:?}");
            std::process::exit(3);
        }
    };
    let generated = started.elapsed();
    std::fs::create_dir_all(&out).expect("output directory");
    for (k, png) in output.images.iter().enumerate() {
        let name = if k == 0 { "image.png".to_string() } else { format!("image_{}.png", k + 1) };
        std::fs::write(out.join(name), png).expect("write output");
    }
    println!("effective_steps {}", output.effective_steps);
    println!("timing load_ms={} generate_ms={}", loaded.as_millis(), generated.as_millis());
}
