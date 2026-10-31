//! The native engine's owned SDXL base implementation on candle (`native-sdxl`).
//!
//! INTERFACE CONTRACT (the server in `native/src` depends on exactly this API):
//! the server validates and verifies requests and bundles; this crate loads a verified
//! sdxl-base Diffusers bundle and executes text-to-image / image-to-image with the semantics of
//! the Python owned pipeline (`src/media_engine/pipeline.py`, `OwnedPipelineBackend`).

use std::path::{Path, PathBuf};

mod clip;
mod config;
mod image;
mod json;
mod nn;
mod pipeline;
pub mod rng;
mod scheduler;
pub mod tokenizer;
#[allow(clippy::needless_range_loop)]
mod unet;
mod unicode;
mod unicode_tables;
#[allow(clippy::needless_range_loop)]
mod vae;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Float32,
    Float16,
    BFloat16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceSpec {
    Cpu,
    Cuda(usize),
}

/// A PEFT-layout LoRA safetensors file merged into the weights at load (as `merge_lora`).
#[derive(Clone, Debug)]
pub struct Adapter {
    pub weights: PathBuf,
    pub scale: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    TextToImage,
    ImageToImage,
}

#[derive(Clone, Debug)]
pub struct Request {
    pub task: Task,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub seed: u64,
    pub steps: usize,
    pub guidance: f64,
    pub width: usize,
    pub height: usize,
    /// Images to produce (1–4); image k uses seed + k (k from 0).
    pub count: usize,
    /// image-to-image only.
    pub strength: f64,
    /// image-to-image only: the PNG bytes of the input (must be width × height).
    pub image_png: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// "encoding", "sampling", "decoding" (the server emits "saving" itself).
    Stage(&'static str),
    /// One per completed sampling step.
    Progress { completed: usize, total: usize },
}

#[derive(Clone, Debug)]
pub struct Output {
    /// One 8-bit RGB PNG per image, in order.
    pub images: Vec<Vec<u8>>,
    pub effective_steps: usize,
}

#[derive(Debug)]
pub enum Error {
    /// The cancellation callback returned true at a checkpoint.
    Cancelled,
    /// A request fact the bundle cannot honor (e.g. "input image size does not match the request").
    InvalidInput(&'static str),
    /// The bundle's configuration is outside what this implementation supports; names the field.
    Unsupported(String),
    /// Anything else (I/O, tensor errors); never shown to clients verbatim.
    Failed(String),
}

pub struct Pipeline {
    _private: (),
    engine: pipeline::Engine,
}

impl Pipeline {
    pub fn load(root: &Path, precision: Precision, device: DeviceSpec, adapters: &[Adapter]) -> Result<Pipeline, Error> {
        Ok(Pipeline { _private: (), engine: pipeline::Engine::load(root, precision, device, adapters)? })
    }

    /// The number of modules merged per adapter, in order (for `loaded_bundle` reporting).
    pub fn merged_modules(&self) -> Vec<usize> {
        self.engine.merged.clone()
    }

    pub fn generate(
        &self,
        request: &Request,
        events: &mut dyn FnMut(Event),
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Output, Error> {
        self.engine.generate(request, events, cancelled)
    }
}

/// image-to-image runs min(int(steps × strength), steps) steps; text-to-image runs `steps`.
pub fn effective_steps(task: Task, steps: usize, strength: f64) -> usize {
    match task {
        Task::TextToImage => steps,
        Task::ImageToImage => ((steps as f64 * strength) as usize).min(steps),
    }
}
