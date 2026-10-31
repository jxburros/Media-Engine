//! transformers' `CLIPTextModel` / `CLIPTextModelWithProjection` (inference, no attention mask
//! beyond the causal one, as the pipeline calls them).

use candle_core::{DType, Device, Tensor};

use crate::config::Config;
use crate::nn::{attention, merge_heads, split_heads, Activation, Fallible, LayerNorm, Linear, Result, Weights};
use crate::Error;

#[derive(Clone, Debug)]
pub struct ClipConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub heads: usize,
    pub layers: usize,
    pub max_positions: usize,
    pub eps: f64,
    pub activation: Activation,
    pub eos_token_id: i64,
    pub projection_dim: usize,
}

impl ClipConfig {
    pub fn parse(c: &Config) -> Result<ClipConfig> {
        if let Some(model_type) = c.opt_str("model_type")? {
            if model_type != "clip_text_model" {
                return Err(c.unsupported("model_type"));
            }
        }
        let activation = match c.str_or("hidden_act", "quick_gelu")?.as_str() {
            "quick_gelu" => Activation::QuickGelu,
            "gelu" => Activation::Gelu,
            "gelu_new" | "gelu_pytorch_tanh" => Activation::GeluTanh,
            _ => return Err(c.unsupported("hidden_act")),
        };
        let config = ClipConfig {
            vocab_size: c.usize_or("vocab_size", 49408)?,
            hidden_size: c.usize_or("hidden_size", 512)?,
            intermediate_size: c.usize_or("intermediate_size", 2048)?,
            heads: c.usize_or("num_attention_heads", 8)?,
            layers: c.usize_or("num_hidden_layers", 12)?,
            max_positions: c.usize_or("max_position_embeddings", 77)?,
            eps: c.f64_or("layer_norm_eps", 1e-5)?,
            activation,
            eos_token_id: c.i64_or("eos_token_id", 49407)?,
            projection_dim: c.usize_or("projection_dim", 512)?,
        };
        if config.heads == 0 || !config.hidden_size.is_multiple_of(config.heads) {
            return Err(c.unsupported("num_attention_heads"));
        }
        if config.layers < 2 {
            return Err(c.unsupported("num_hidden_layers"));
        }
        Ok(config)
    }
}

struct EncoderLayer {
    norm1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

pub struct ClipText {
    config: ClipConfig,
    token_embedding: Tensor,
    position_embedding: Tensor,
    layers: Vec<EncoderLayer>,
    final_norm: Option<LayerNorm>,
    projection: Option<Linear>,
}

pub struct ClipOutput {
    /// `hidden_states[-2]`: the penultimate layer's output (before the final layer norm).
    pub penultimate: Tensor,
    /// `text_embeds` (with projection) for the projection model.
    pub pooled: Option<Tensor>,
}

impl ClipText {
    /// `with_projection` loads `CLIPTextModelWithProjection` (and computes the pooled projection);
    /// otherwise `CLIPTextModel`, whose last layer and final norm are loaded (strict state dict)
    /// but never needed for the penultimate hidden state.
    pub fn load(w: &mut Weights, config: ClipConfig, with_projection: bool) -> Result<ClipText> {
        let c = &config;
        let h = c.hidden_size;
        let p = "text_model";
        let token_embedding = w.take(&format!("{p}.embeddings.token_embedding.weight"), &[c.vocab_size, h])?;
        let position_embedding = w.take(&format!("{p}.embeddings.position_embedding.weight"), &[c.max_positions, h])?;
        let mut layers = Vec::new();
        for i in 0..c.layers {
            let l = format!("{p}.encoder.layers.{i}");
            layers.push(EncoderLayer {
                norm1: LayerNorm::load(w, &format!("{l}.layer_norm1"), h, c.eps)?,
                q: Linear::load(w, &format!("{l}.self_attn.q_proj"), h, h, true)?,
                k: Linear::load(w, &format!("{l}.self_attn.k_proj"), h, h, true)?,
                v: Linear::load(w, &format!("{l}.self_attn.v_proj"), h, h, true)?,
                out: Linear::load(w, &format!("{l}.self_attn.out_proj"), h, h, true)?,
                norm2: LayerNorm::load(w, &format!("{l}.layer_norm2"), h, c.eps)?,
                fc1: Linear::load(w, &format!("{l}.mlp.fc1"), h, c.intermediate_size, true)?,
                fc2: Linear::load(w, &format!("{l}.mlp.fc2"), c.intermediate_size, h, true)?,
            });
        }
        let final_norm = Some(LayerNorm::load(w, &format!("{p}.final_layer_norm"), h, c.eps)?);
        let projection =
            if with_projection { Some(Linear::load(w, "text_projection", h, c.projection_dim, false)?) } else { None };
        Ok(ClipText { config, token_embedding, position_embedding, layers, final_norm, projection })
    }

    fn layer(&self, layer: &EncoderLayer, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let heads = self.config.heads;
        let normed = layer.norm1.forward(x)?;
        let q = split_heads(&layer.q.forward(&normed)?, heads)?;
        let k = split_heads(&layer.k.forward(&normed)?, heads)?;
        let v = split_heads(&layer.v.forward(&normed)?, heads)?;
        let attended = layer.out.forward(&merge_heads(&attention(&q, &k, &v, Some(mask))?)?)?;
        let x = (x + attended).ok_or_failed()?;
        let hidden =
            layer.fc2.forward(&self.config.activation.forward(&layer.fc1.forward(&layer.norm2.forward(&x)?)?)?)?;
        (x + hidden).ok_or_failed()
    }

    /// Encode one row of token ids.
    pub fn forward(&self, ids: &[u32], device: &Device) -> Result<ClipOutput> {
        let len = ids.len();
        if len > self.config.max_positions {
            return Err(Error::Unsupported("max_position_embeddings".into()));
        }
        if let Some(&bad) = ids.iter().find(|&&i| i as usize >= self.config.vocab_size) {
            return Err(Error::Failed(format!("token id {bad} outside the text encoder vocabulary")));
        }
        let ids_t = Tensor::new(ids, device).ok_or_failed()?;
        let dtype = self.token_embedding.dtype();
        let tokens = self.token_embedding.index_select(&ids_t, 0).ok_or_failed()?;
        let positions = self.position_embedding.narrow(0, 0, len).ok_or_failed()?;
        let mut x = (tokens + positions).and_then(|t| t.unsqueeze(0)).ok_or_failed()?;
        // The causal mask (torch.finfo(dtype).min above the diagonal).
        let min = match dtype {
            DType::F16 => -65504.0f32,
            DType::BF16 => -3.389_531_4e38,
            _ => f32::MIN,
        };
        let mask: Vec<f32> = (0..len * len).map(|i| if i % len > i / len { min } else { 0.0 }).collect();
        let mask = Tensor::from_vec(mask, (1, 1, len, len), device).and_then(|t| t.to_dtype(dtype)).ok_or_failed()?;
        let last = self.layers.len() - 1;
        let needed = if self.projection.is_some() { self.layers.len() } else { last };
        let mut penultimate = None;
        for (i, layer) in self.layers.iter().enumerate().take(needed) {
            if i == last {
                penultimate = Some(x.clone());
            }
            x = self.layer(layer, &x, &mask)?;
        }
        let penultimate = match penultimate {
            Some(p) => p,
            None => x.clone(),
        };
        let pooled = match &self.projection {
            None => None,
            Some(projection) => {
                let normed = self.final_norm.as_ref().unwrap().forward(&x)?;
                // eos_token_id == 2 is the legacy configuration: pool at the highest token id.
                let position = if self.config.eos_token_id == 2 {
                    argmax_first(ids)
                } else {
                    ids.iter().position(|&i| i as i64 == self.config.eos_token_id).unwrap_or(0)
                };
                let row = normed.narrow(1, position, 1).and_then(|t| t.squeeze(1)).ok_or_failed()?;
                Some(projection.forward(&row)?)
            }
        };
        Ok(ClipOutput { penultimate, pooled })
    }
}

fn argmax_first(ids: &[u32]) -> usize {
    let mut best = 0;
    for (i, &v) in ids.iter().enumerate() {
        if v > ids[best] {
            best = i;
        }
    }
    best
}
