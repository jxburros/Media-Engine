//! Loading and execution with the semantics of `OwnedPipelineBackend` (src/media_engine/pipeline.py).

use std::collections::BTreeMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};

use crate::clip::{ClipConfig, ClipText};
use crate::config::Config;
use crate::json::Value;
use crate::nn::{add_scalar, div_scalar, mul_scalar, Fallible, Result, Weights};
use crate::rng::{round_bf16, round_f16, RandnDtype, TorchGenerator};
use crate::scheduler::{EulerConfig, Prediction};
use crate::tokenizer::ClipTokenizer;
use crate::unet::{UNet, UNetConfig};
use crate::vae::{Vae, VaeConfig};
use crate::{effective_steps, Adapter, DeviceSpec, Error, Event, Output, Precision, Request, Task};

pub struct Engine {
    device: Device,
    dtype: DType,
    vae_dtype: DType,
    unet: UNet,
    vae: Vae,
    text_encoder: ClipText,
    text_encoder_2: ClipText,
    tokenizer: ClipTokenizer,
    tokenizer_2: ClipTokenizer,
    scheduler: EulerConfig,
    force_zeros_for_empty_prompt: bool,
    pub merged: Vec<usize>,
}

fn device_for(spec: DeviceSpec) -> Result<Device> {
    match spec {
        DeviceSpec::Cpu => Ok(Device::Cpu),
        DeviceSpec::Cuda(index) => Device::new_cuda(index).map_err(|_| Error::Unsupported("device".into())),
    }
}

fn randn_dtype(dtype: DType) -> RandnDtype {
    match dtype {
        DType::F16 => RandnDtype::Float16,
        DType::BF16 => RandnDtype::BFloat16,
        _ => RandnDtype::Float32,
    }
}

fn round_to(dtype: DType, v: f32) -> f32 {
    match dtype {
        DType::F16 => round_f16(v),
        DType::BF16 => round_bf16(v),
        _ => v,
    }
}

/// `merge_lora`: fold scale × alpha/rank × B·A into the targets' weights; returns the number of
/// modules merged.
fn merge_lora(targets: &mut [(&str, &mut Weights)], path: &Path, scale: f64) -> Result<usize> {
    let state = candle_core::safetensors::load(path, &Device::Cpu)
        .map_err(|e| Error::Failed(format!("adapter weights: {e}")))?;
    #[derive(Default)]
    struct Parts {
        down: Option<Tensor>,
        up: Option<Tensor>,
        alpha: Option<Tensor>,
    }
    let mut groups: BTreeMap<(String, String), Parts> = BTreeMap::new();
    for (name, tensor) in state {
        let (target, rest) = ["unet.", "text_encoder.", "text_encoder_2."]
            .iter()
            .find_map(|p| name.strip_prefix(p).map(|rest| (p.trim_end_matches('.'), rest)))
            .ok_or_else(|| Error::Unsupported("adapter key layout".into()))?;
        let (module, part) = [(".lora_A.weight", 0), (".lora_B.weight", 1), (".alpha", 2)]
            .iter()
            .find_map(|(suffix, part)| rest.strip_suffix(suffix).map(|m| (m, *part)))
            .filter(|(m, _)| !m.is_empty())
            .ok_or_else(|| Error::Unsupported("adapter key layout".into()))?;
        let entry = groups.entry((target.to_string(), module.to_string())).or_default();
        match part {
            0 => entry.down = Some(tensor),
            1 => entry.up = Some(tensor),
            _ => entry.alpha = Some(tensor),
        }
    }
    let mut merged = 0;
    for ((target, module), parts) in groups {
        let (down, up) = match (parts.down, parts.up) {
            (Some(d), Some(u)) => (d, u),
            _ => return Err(Error::Failed("adapter module lacks lora_A or lora_B".into())),
        };
        let weights = targets
            .iter_mut()
            .find(|(name, _)| *name == target)
            .map(|(_, w)| w)
            .ok_or_else(|| Error::Failed("adapter target missing".into()))?;
        let weight = weights
            .get_mut(&format!("{module}.weight"))
            .ok_or_else(|| Error::Failed("adapter module is not in the bundle".into()))?;
        let down = down.to_dtype(DType::F32).ok_or_failed()?;
        let up = up.to_dtype(DType::F32).ok_or_failed()?;
        let rank = down.dims()[0];
        let alpha = match parts.alpha {
            Some(a) => {
                if a.elem_count() != 1 {
                    return Err(Error::Failed("adapter alpha is not a scalar".into()));
                }
                a.to_dtype(DType::F64).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f64>()).ok_or_failed()?[0]
            }
            None => rank as f64,
        };
        let delta = up.flatten_from(1).ok_or_failed()?.matmul(&down.flatten_from(1).ok_or_failed()?).ok_or_failed()?;
        if delta.elem_count() != weight.elem_count() {
            return Err(Error::Failed("adapter shape does not match the bundle".into()));
        }
        let delta = delta.reshape(weight.dims()).ok_or_failed()?;
        let coefficient = (scale * alpha / rank as f64) as f32;
        let delta = mul_scalar(&delta, coefficient)?
            .to_dtype(weight.dtype())
            .and_then(|d| d.to_device(weight.device()))
            .ok_or_failed()?;
        *weight = (&*weight + delta).ok_or_failed()?;
        merged += 1;
    }
    Ok(merged)
}

/// The cross-component dimensions SDXL conditioning needs: the context is the two penultimate
/// states side by side, the addition embedding takes encoder 2's projection plus six time ids.
fn check_conditioning(unet: &UNetConfig, vae: &VaeConfig, te: &ClipConfig, te2: &ClipConfig) -> Result<()> {
    let context = te.hidden_size + te2.hidden_size;
    if unet.cross_attention_dim.iter().any(|&d| d != context) {
        return Err(Error::Unsupported("unet.cross_attention_dim".into()));
    }
    if unet.projection_class_embeddings_input_dim != 6 * unet.addition_time_embed_dim + te2.projection_dim {
        return Err(Error::Unsupported("unet.projection_class_embeddings_input_dim".into()));
    }
    if vae.latent_channels != unet.in_channels || unet.out_channels != unet.in_channels {
        return Err(Error::Unsupported("unet.in_channels".into()));
    }
    Ok(())
}

impl Engine {
    pub fn load(root: &Path, precision: Precision, device: DeviceSpec, adapters: &[Adapter]) -> Result<Engine> {
        let device = device_for(device)?;
        let dtype = match precision {
            Precision::Float32 => DType::F32,
            Precision::Float16 => DType::F16,
            Precision::BFloat16 => DType::BF16,
        };
        let index = Config::read(&root.join("model_index.json"), "model_index")?;
        // `bool(index.get("force_zeros_for_empty_prompt", True))`: Python truthiness, null is false.
        let force_zeros_for_empty_prompt = match index.value.get("force_zeros_for_empty_prompt") {
            None => true,
            Some(Value::Null) => false,
            Some(Value::Bool(b)) => *b,
            Some(Value::Number(_, n)) => *n != 0.0,
            Some(Value::String(s)) => !s.is_empty(),
            Some(Value::Array(a)) => !a.is_empty(),
            Some(Value::Object(o)) => !o.is_empty(),
        };
        let unet_config = UNetConfig::parse(&Config::read(&root.join("unet/config.json"), "unet")?)?;
        let vae_config = VaeConfig::parse(&Config::read(&root.join("vae/config.json"), "vae")?)?;
        let te_config = ClipConfig::parse(&Config::read(&root.join("text_encoder/config.json"), "text_encoder")?)?;
        let te2_config = ClipConfig::parse(&Config::read(&root.join("text_encoder_2/config.json"), "text_encoder_2")?)?;
        let scheduler = EulerConfig::parse(&Config::read(&root.join("scheduler/scheduler_config.json"), "scheduler")?)?;
        let tokenizer = ClipTokenizer::from_dir(&root.join("tokenizer"))?;
        let tokenizer_2 = ClipTokenizer::from_dir(&root.join("tokenizer_2"))?;
        if tokenizer.max_length != tokenizer_2.max_length {
            return Err(Error::Unsupported("tokenizer_2.model_max_length".into()));
        }
        if tokenizer.max_length > te_config.max_positions {
            return Err(Error::Unsupported("tokenizer.model_max_length".into()));
        }
        if tokenizer_2.max_length > te2_config.max_positions {
            return Err(Error::Unsupported("tokenizer_2.model_max_length".into()));
        }
        check_conditioning(&unet_config, &vae_config, &te_config, &te2_config)?;

        let mut unet_w = Weights::load(&root.join("unet"), "unet", dtype, &device)?;
        // The VAE is kept float32 for float16 when the bundle asks for an upcast (weights are cast
        // to the load precision first, then up, as the owned path does).
        let vae_dtype = if dtype == DType::F16 && vae_config.force_upcast { DType::F32 } else { dtype };
        let mut vae_w = Weights::load(&root.join("vae"), "vae", dtype, &device)?;
        if vae_dtype != dtype {
            vae_w = vae_w.cast(vae_dtype)?;
        }
        let mut te_w = Weights::load(&root.join("text_encoder"), "text_encoder", dtype, &device)?;
        let mut te2_w = Weights::load(&root.join("text_encoder_2"), "text_encoder_2", dtype, &device)?;
        let mut merged = Vec::new();
        for adapter in adapters {
            let mut targets = [("unet", &mut unet_w), ("text_encoder", &mut te_w), ("text_encoder_2", &mut te2_w)];
            merged.push(merge_lora(&mut targets, &adapter.weights, adapter.scale)?);
        }
        let unet = UNet::load(&mut unet_w, unet_config)?;
        unet_w.finish()?;
        let vae = Vae::load(&mut vae_w, vae_config)?;
        vae_w.finish()?;
        let text_encoder = ClipText::load(&mut te_w, te_config, false)?;
        te_w.finish()?;
        let text_encoder_2 = ClipText::load(&mut te2_w, te2_config, true)?;
        te2_w.finish()?;
        Ok(Engine {
            device,
            dtype,
            vae_dtype,
            unet,
            vae,
            text_encoder,
            text_encoder_2,
            tokenizer,
            tokenizer_2,
            scheduler,
            force_zeros_for_empty_prompt,
            merged,
        })
    }

    /// `encode_text`: (penultimate states of both encoders concatenated, encoder 2's projection).
    fn encode_text(&self, text: &str) -> Result<(Tensor, Tensor)> {
        let ids = self.tokenizer.encode(text)?;
        let ids_2 = self.tokenizer_2.encode(text)?;
        let first = self.text_encoder.forward(&ids, &self.device)?;
        let second = self.text_encoder_2.forward(&ids_2, &self.device)?;
        let hidden = Tensor::cat(&[&first.penultimate, &second.penultimate], 2).ok_or_failed()?;
        Ok((hidden, second.pooled.unwrap()))
    }

    /// `conditioning`: (context, pooled, time ids per row, classifier-free guidance on).
    fn conditioning(&self, request: &Request) -> Result<(Tensor, Tensor, Vec<Vec<f32>>, bool)> {
        let (embeds, pooled) = self.encode_text(&request.prompt)?;
        let (h, w) = (request.height as f32, request.width as f32);
        let ids: Vec<f32> = [h, w, 0.0, 0.0, h, w].iter().map(|&v| round_to(self.dtype, v)).collect();
        if request.guidance <= 1.0 {
            return Ok((embeds, pooled, vec![ids], false));
        }
        let (negative, negative_pooled) = match &request.negative_prompt {
            None if self.force_zeros_for_empty_prompt => {
                (embeds.zeros_like().ok_or_failed()?, pooled.zeros_like().ok_or_failed()?)
            }
            other => self.encode_text(other.as_deref().unwrap_or(""))?,
        };
        Ok((
            Tensor::cat(&[&negative, &embeds], 0).ok_or_failed()?,
            Tensor::cat(&[&negative_pooled, &pooled], 0).ok_or_failed()?,
            vec![ids.clone(), ids],
            true,
        ))
    }

    fn randn(&self, generator: &mut TorchGenerator, shape: &[usize], dtype: DType) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let values = generator.randn(n, randn_dtype(dtype));
        Tensor::from_vec(values, shape, &self.device).and_then(|t| t.to_dtype(dtype)).ok_or_failed()
    }

    /// `encode_image`: the posterior sample (drawn first from the image's generator) × scaling.
    fn encode_image(&self, png: &[u8], width: usize, height: usize, generator: &mut TorchGenerator) -> Result<Tensor> {
        let (w, h, rgb) = crate::image::decode_rgb(png)?;
        if (w, h) != (width, height) {
            return Err(Error::InvalidInput("input image size does not match the request"));
        }
        let plane = w * h;
        let mut chw = vec![0f32; 3 * plane];
        for i in 0..plane {
            for c in 0..3 {
                let v = rgb[i * 3 + c] as f32 / 255.0f32;
                chw[c * plane + i] = v * 2.0f32 - 1.0f32;
            }
        }
        let image = Tensor::from_vec(chw, (1, 3, h, w), &self.device)
            .and_then(|t| t.to_dtype(self.vae_dtype))
            .ok_or_failed()?;
        let (mean, std) = self.vae.encode(&image)?;
        let noise = self.randn(generator, mean.dims(), self.vae_dtype)?;
        let sample = (&mean + (&std * noise).ok_or_failed()?).ok_or_failed()?;
        mul_scalar(&sample, self.vae.config.scaling_factor as f32)?.to_dtype(self.dtype).ok_or_failed()
    }

    /// `decode`: one latent (1, C, h, w) to PNG bytes.
    fn decode(&self, latent: &Tensor) -> Result<Vec<u8>> {
        let latent =
            div_scalar(&latent.to_dtype(self.vae_dtype).ok_or_failed()?, self.vae.config.scaling_factor as f32)?;
        let image = self.vae.decode(&latent)?;
        let image = add_scalar(&div_scalar(&image, 2.0)?, 0.5)?.clamp(0f32, 1f32).ok_or_failed()?;
        let (_, _, h, w) = image.dims4().ok_or_failed()?;
        let values = image
            .to_dtype(DType::F32)
            .and_then(|t| t.to_device(&Device::Cpu))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .ok_or_failed()?;
        crate::image::encode_rgb(w, h, &crate::image::to_bytes(&values, w, h))
    }

    pub fn generate(
        &self,
        request: &Request,
        events: &mut dyn FnMut(Event),
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Output> {
        let check = || if cancelled() { Err(Error::Cancelled) } else { Ok(()) };
        let factor = self.vae.config.scale_factor();
        if request.count == 0 || request.count > 4 {
            return Err(Error::InvalidInput("count must be between 1 and 4"));
        }
        if request.task == Task::ImageToImage && request.count != 1 {
            return Err(Error::InvalidInput("count applies to text-to-image only"));
        }
        if request.width == 0
            || request.height == 0
            || !request.width.is_multiple_of(factor)
            || !request.height.is_multiple_of(factor)
        {
            return Err(Error::InvalidInput("width and height must be multiples of the VAE scale factor"));
        }
        if request.steps == 0 {
            return Err(Error::InvalidInput("steps must be positive"));
        }
        let total = effective_steps(request.task, request.steps, request.strength);
        if request.task == Task::ImageToImage {
            if request.image_png.is_none() {
                return Err(Error::InvalidInput("image-to-image needs an input image"));
            }
            if total == 0 {
                return Err(Error::InvalidInput("strength leaves no denoising steps"));
            }
        }
        check()?;
        events(Event::Stage("encoding"));
        let (mut embeds, mut pooled, mut time_ids, cfg) = self.conditioning(request)?;
        let count = request.count;
        let mut generators: Vec<TorchGenerator> =
            (0..count).map(|k| TorchGenerator::new(request.seed.wrapping_add(k as u64))).collect();
        let schedule = self.scheduler.schedule(request.steps)?;
        let shape = [1, self.unet.config.in_channels, request.height / factor, request.width / factor];
        let (timesteps, mut step_index, mut latents) = if request.task == Task::ImageToImage {
            let start = request.steps - total;
            let initial = self.encode_image(
                request.image_png.as_ref().unwrap(),
                request.width,
                request.height,
                &mut generators[0],
            )?;
            let noise = self.randn(&mut generators[0], &shape, self.dtype)?;
            let latents =
                (initial + mul_scalar(&noise, round_to(self.dtype, schedule.sigmas[start]))?).ok_or_failed()?;
            (schedule.timesteps[start..].to_vec(), start, latents)
        } else {
            let noise: Vec<Tensor> =
                generators.iter_mut().map(|g| self.randn(g, &shape, self.dtype)).collect::<Result<_>>()?;
            let noise = Tensor::cat(&noise, 0).ok_or_failed()?;
            let first = schedule.index_for_timestep(schedule.timesteps[0]);
            (schedule.timesteps.clone(), first, mul_scalar(&noise, schedule.init_noise_sigma)?)
        };
        if count > 1 {
            // Conditioning rows follow the latent batch: [negative × N, positive × N].
            let batch = |t: &Tensor| -> Result<Tensor> {
                let parts = if cfg { t.chunk(2, 0).ok_or_failed()? } else { vec![t.clone()] };
                let mut rows = Vec::new();
                for part in &parts {
                    for _ in 0..count {
                        rows.push(part.clone());
                    }
                }
                Tensor::cat(&rows, 0).ok_or_failed()
            };
            embeds = batch(&embeds)?;
            pooled = batch(&pooled)?;
            let halves = if cfg { 2 } else { 1 };
            time_ids = (0..halves * count).map(|_| time_ids[0].clone()).collect();
        }
        check()?;
        events(Event::Stage("sampling"));
        let steps = timesteps.len();
        for (index, &t) in timesteps.iter().enumerate() {
            check()?;
            if step_index + 1 >= schedule.sigmas.len() {
                return Err(Error::Failed("scheduler step outside the schedule".into()));
            }
            let sigma = schedule.sigmas[step_index];
            let model_input = if cfg { Tensor::cat(&[&latents, &latents], 0).ok_or_failed()? } else { latents.clone() };
            let model_input = div_scalar(&model_input, (sigma * sigma + 1.0).sqrt())?;
            let mut prediction = self.unet.forward(&model_input, t, &embeds, &pooled, &time_ids)?;
            if cfg {
                let parts = prediction.chunk(2, 0).ok_or_failed()?;
                let diff = (&parts[1] - &parts[0]).ok_or_failed()?;
                prediction = (&parts[0] + mul_scalar(&diff, request.guidance as f32)?).ok_or_failed()?;
            }
            latents = self.step(&prediction, &latents, &schedule.sigmas, step_index, schedule.prediction)?;
            step_index += 1;
            events(Event::Progress { completed: index + 1, total: steps });
        }
        check()?;
        events(Event::Stage("decoding"));
        let mut images = Vec::with_capacity(count);
        for k in 0..latents.dims()[0] {
            check()?;
            images.push(self.decode(&latents.narrow(0, k, 1).ok_or_failed()?)?);
        }
        check()?;
        Ok(Output { images, effective_steps: steps })
    }

    /// `EulerDiscreteScheduler.step` with s_churn = 0.
    fn step(
        &self,
        output: &Tensor,
        sample: &Tensor,
        sigmas: &[f32],
        i: usize,
        prediction: Prediction,
    ) -> Result<Tensor> {
        let dtype = output.dtype();
        let sample = sample.to_dtype(DType::F32).ok_or_failed()?;
        let sigma = sigmas[i];
        let sigma_hat = sigma * (0.0f32 + 1.0);
        let original = match prediction {
            Prediction::Epsilon => {
                (&sample - mul_scalar(output, sigma_hat)?.to_dtype(DType::F32).ok_or_failed()?).ok_or_failed()?
            }
            Prediction::VPrediction => {
                let denom = sigma * sigma + 1.0;
                let a = mul_scalar(output, -sigma / denom.sqrt())?.to_dtype(DType::F32).ok_or_failed()?;
                (a + div_scalar(&sample, denom)?).ok_or_failed()?
            }
        };
        let derivative = div_scalar(&(&sample - original).ok_or_failed()?, sigma_hat)?;
        let dt = sigmas[i + 1] - sigma_hat;
        (sample + mul_scalar(&derivative, dt)?).and_then(|t| t.to_dtype(dtype)).ok_or_failed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;
    use crate::unet::BlockKind;

    fn config(component: &str, text: &str) -> Config {
        Config { component: component.into(), value: json::parse(text).unwrap() }
    }

    const UNET: &str = r#"{"_class_name": "UNet2DConditionModel", "_diffusers_version": "0.19.0.dev0", "act_fn": "silu",
        "addition_embed_type": "text_time", "addition_embed_type_num_heads": 64, "addition_time_embed_dim": 256,
        "attention_head_dim": [5, 10, 20], "block_out_channels": [320, 640, 1280], "center_input_sample": false,
        "class_embed_type": null, "class_embeddings_concat": false, "conv_in_kernel": 3, "conv_out_kernel": 3,
        "cross_attention_dim": 2048, "cross_attention_norm": null,
        "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"], "downsample_padding": 1,
        "dual_cross_attention": false, "encoder_hid_dim": null, "encoder_hid_dim_type": null, "flip_sin_to_cos": true,
        "freq_shift": 0, "in_channels": 4, "layers_per_block": 2, "mid_block_only_cross_attention": null,
        "mid_block_scale_factor": 1, "mid_block_type": "UNetMidBlock2DCrossAttn", "norm_eps": 1e-05,
        "norm_num_groups": 32, "num_attention_heads": null, "num_class_embeds": null, "only_cross_attention": false,
        "out_channels": 4, "projection_class_embeddings_input_dim": 2816, "resnet_out_scale_factor": 1.0,
        "resnet_skip_time_act": false, "resnet_time_scale_shift": "default", "sample_size": 128,
        "time_cond_proj_dim": null, "time_embedding_act_fn": null, "time_embedding_dim": null,
        "time_embedding_type": "positional", "timestep_post_act": null, "transformer_layers_per_block": [1, 2, 10],
        "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"], "upcast_attention": null,
        "use_linear_projection": true}"#;
    const VAE: &str = r#"{"_class_name": "AutoencoderKL", "act_fn": "silu", "block_out_channels": [128, 256, 512, 512],
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        "force_upcast": true, "in_channels": 3, "latent_channels": 4, "layers_per_block": 2, "norm_num_groups": 32,
        "out_channels": 3, "sample_size": 1024, "scaling_factor": 0.13025,
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"]}"#;
    const TEXT_ENCODER: &str = r#"{"architectures": ["CLIPTextModel"], "attention_dropout": 0.0, "bos_token_id": 0,
        "dropout": 0.0, "eos_token_id": 2, "hidden_act": "quick_gelu", "hidden_size": 768, "initializer_factor": 1.0,
        "initializer_range": 0.02, "intermediate_size": 3072, "layer_norm_eps": 1e-05, "max_position_embeddings": 77,
        "model_type": "clip_text_model", "num_attention_heads": 12, "num_hidden_layers": 12, "pad_token_id": 1,
        "projection_dim": 768, "torch_dtype": "float16", "vocab_size": 49408}"#;
    const TEXT_ENCODER_2: &str = r#"{"architectures": ["CLIPTextModelWithProjection"], "attention_dropout": 0.0,
        "bos_token_id": 0, "dropout": 0.0, "eos_token_id": 2, "hidden_act": "gelu", "hidden_size": 1280,
        "initializer_factor": 1.0, "initializer_range": 0.02, "intermediate_size": 5120, "layer_norm_eps": 1e-05,
        "max_position_embeddings": 77, "model_type": "clip_text_model", "num_attention_heads": 20,
        "num_hidden_layers": 32, "pad_token_id": 1, "projection_dim": 1280, "torch_dtype": "float16", "vocab_size": 49408}"#;
    const SCHEDULER: &str = r#"{"_class_name": "EulerDiscreteScheduler", "beta_end": 0.012, "beta_schedule": "scaled_linear",
        "beta_start": 0.00085, "clip_sample": false, "interpolation_type": "linear", "num_train_timesteps": 1000,
        "prediction_type": "epsilon", "sample_max_value": 1.0, "set_alpha_to_one": false, "skip_prk_steps": true,
        "steps_offset": 1, "timestep_spacing": "leading", "trained_betas": null, "use_karras_sigmas": false}"#;

    #[test]
    fn pipeline_is_shareable_across_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<crate::Pipeline>();
    }

    /// The SDXL base component configs (as published in its Diffusers layout) parse into its architecture.
    #[test]
    fn sdxl_base_configs_parse() {
        let unet = UNetConfig::parse(&config("unet", UNET)).unwrap();
        assert_eq!(unet.down_blocks, vec![BlockKind::Plain, BlockKind::CrossAttn, BlockKind::CrossAttn]);
        assert_eq!(unet.up_blocks, vec![BlockKind::CrossAttn, BlockKind::CrossAttn, BlockKind::Plain]);
        assert_eq!(unet.heads, vec![5, 10, 20]);
        assert_eq!(unet.transformer_layers, vec![vec![1, 1], vec![2, 2], vec![10, 10]]);
        assert_eq!(unet.reverse_transformer_layers, vec![vec![10, 10, 10], vec![2, 2, 2], vec![1, 1, 1]]);
        assert_eq!((unet.time_embed_dim, unet.addition_time_embed_dim), (1280, 256));
        let vae = VaeConfig::parse(&config("vae", VAE)).unwrap();
        assert_eq!((vae.scale_factor(), vae.force_upcast, vae.scaling_factor), (8, true, 0.13025));
        let te = ClipConfig::parse(&config("text_encoder", TEXT_ENCODER)).unwrap();
        let te2 = ClipConfig::parse(&config("text_encoder_2", TEXT_ENCODER_2)).unwrap();
        assert!(matches!(te.activation, crate::nn::Activation::QuickGelu));
        assert!(matches!(te2.activation, crate::nn::Activation::Gelu));
        assert_eq!((te2.layers, te2.heads, te2.projection_dim, te2.eos_token_id), (32, 20, 1280, 2));
        check_conditioning(&unet, &vae, &te, &te2).unwrap();
        let schedule = EulerConfig::parse(&config("scheduler", SCHEDULER)).unwrap().schedule(30).unwrap();
        assert_eq!(schedule.timesteps[..3], [958.0, 925.0, 892.0]);
        assert_eq!(*schedule.timesteps.last().unwrap(), 1.0);
    }

    fn unsupported(component: &str, base: &str, key: &str, value: &str) -> String {
        let mut text = base.trim_end().trim_end_matches('}').to_string();
        text.push_str(&format!(r#", "{key}": {value}}}"#));
        let c = config(component, &text);
        let result = match component {
            "unet" => UNetConfig::parse(&c).map(|_| ()),
            "vae" => VaeConfig::parse(&c).map(|_| ()),
            _ => ClipConfig::parse(&c).map(|_| ()),
        };
        match result {
            Err(Error::Unsupported(field)) => field,
            other => panic!("{key}={value} was accepted: {:?}", other),
        }
    }

    /// Configuration outside the executed subset is refused, naming the field.
    #[test]
    fn unsupported_configs_name_the_field() {
        for (key, value) in [
            ("attention_type", r#""gated""#),
            ("resnet_time_scale_shift", r#""scale_shift""#),
            ("num_attention_heads", "8"),
            ("addition_embed_type", "null"),
            ("class_embed_type", r#""timestep""#),
            ("time_embedding_type", r#""fourier""#),
            ("down_block_types", r#"["DownBlock2D", "SimpleCrossAttnDownBlock2D", "CrossAttnDownBlock2D"]"#),
            ("mid_block_type", "null"),
            ("only_cross_attention", "true"),
            ("transformer_layers_per_block", "[1, 2]"),
        ] {
            assert_eq!(unsupported("unet", UNET, key, value), format!("unet.{key}"));
        }
        assert_eq!(unsupported("vae", VAE, "shift_factor", "0.1159"), "vae.shift_factor");
        assert_eq!(unsupported("vae", VAE, "latents_mean", "[0.1, 0.2, 0.3, 0.4]"), "vae.latents_mean");
        assert_eq!(unsupported("text_encoder", TEXT_ENCODER, "hidden_act", r#""relu""#), "text_encoder.hidden_act");
        let te = ClipConfig::parse(&config("text_encoder", TEXT_ENCODER)).unwrap();
        let te2 = ClipConfig::parse(&config("text_encoder_2", TEXT_ENCODER_2)).unwrap();
        let mut unet = UNetConfig::parse(&config("unet", UNET)).unwrap();
        unet.projection_class_embeddings_input_dim = 2560;
        let vae = VaeConfig::parse(&config("vae", VAE)).unwrap();
        assert!(
            matches!(check_conditioning(&unet, &vae, &te, &te2), Err(Error::Unsupported(f)) if f == "unet.projection_class_embeddings_input_dim")
        );
    }
}
