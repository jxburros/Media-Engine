//! diffusers' `UNet2DConditionModel` for the SDXL configuration family (text_time addition
//! embedding, CrossAttn/plain down and up blocks, `UNetMidBlock2DCrossAttn`).

use candle_core::{Device, Tensor};

use crate::config::Config;
use crate::json::Value;
use crate::nn::{
    attention, merge_heads, split_heads, upsample_nearest, Activation, Conv2d, Fallible, GroupNorm, LayerNorm, Linear,
    Result, Weights,
};
use crate::Error;

#[derive(Clone, Debug, PartialEq)]
pub enum BlockKind {
    Plain,
    CrossAttn,
}

#[derive(Clone, Debug)]
pub struct UNetConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub center_input_sample: bool,
    pub flip_sin_to_cos: bool,
    pub freq_shift: f64,
    pub down_blocks: Vec<BlockKind>,
    pub up_blocks: Vec<BlockKind>,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: Vec<usize>,
    pub downsample_padding: usize,
    pub mid_block_scale_factor: f64,
    pub norm_num_groups: usize,
    pub norm_eps: f64,
    pub cross_attention_dim: Vec<usize>,
    /// Per down block, per resnet layer.
    pub transformer_layers: Vec<Vec<usize>>,
    /// Per up block, per resnet layer.
    pub reverse_transformer_layers: Vec<Vec<usize>>,
    /// `attention_head_dim`, which this model family uses as the number of heads.
    pub heads: Vec<usize>,
    pub use_linear_projection: bool,
    pub addition_time_embed_dim: usize,
    pub projection_class_embeddings_input_dim: usize,
    pub time_embed_dim: usize,
    pub conv_in_kernel: usize,
    pub conv_out_kernel: usize,
}

fn nested_layers(c: &Config, key: &str, value: Option<&Value>, layers: &[usize]) -> Result<Vec<Vec<usize>>> {
    let n = layers.len();
    let entry = |v: &Value, count: usize| -> Result<Vec<usize>> {
        match v {
            Value::Array(items) => {
                if items.len() != count {
                    return Err(c.unsupported(key));
                }
                items
                    .iter()
                    .map(|x| x.as_i64().filter(|i| *i > 0).map(|i| i as usize).ok_or_else(|| c.unsupported(key)))
                    .collect()
            }
            _ => v.as_i64().filter(|i| *i > 0).map(|i| vec![i as usize; count]).ok_or_else(|| c.unsupported(key)),
        }
    };
    match value {
        None => Ok(layers.iter().map(|&l| vec![1; l]).collect()),
        Some(Value::Array(items)) => {
            if items.len() != n {
                return Err(c.unsupported(key));
            }
            items.iter().zip(layers).map(|(v, &l)| entry(v, l)).collect()
        }
        Some(v) => {
            let t = v.as_i64().filter(|i| *i > 0).ok_or_else(|| c.unsupported(key))? as usize;
            Ok(layers.iter().map(|&l| vec![t; l]).collect())
        }
    }
}

impl UNetConfig {
    pub fn parse(c: &Config) -> Result<UNetConfig> {
        // Fields whose non-default values this implementation does not execute.
        for key in [
            "encoder_hid_dim",
            "encoder_hid_dim_type",
            "num_attention_heads",
            "class_embed_type",
            "num_class_embeds",
            "time_embedding_act_fn",
            "timestep_post_act",
            "time_cond_proj_dim",
            "cross_attention_norm",
        ] {
            c.require_null(key)?;
        }
        // Structural fields have no meaning as null (the model class would fail on them).
        for key in [
            "block_out_channels",
            "layers_per_block",
            "attention_head_dim",
            "cross_attention_dim",
            "transformer_layers_per_block",
            "down_block_types",
            "up_block_types",
            "in_channels",
            "out_channels",
        ] {
            if matches!(c.value.get(key), Some(Value::Null)) {
                return Err(c.unsupported(key));
            }
        }
        c.require_bool("dual_cross_attention", false)?;
        c.require_bool("resnet_skip_time_act", false)?;
        c.require_bool("class_embeddings_concat", false)?;
        c.require_bool("mid_block_only_cross_attention", false)?;
        match c.get("only_cross_attention") {
            None | Some(Value::Bool(false)) => {}
            Some(Value::Array(items)) if items.iter().all(|v| v.as_bool() == Some(false)) => {}
            Some(_) => return Err(c.unsupported("only_cross_attention")),
        }
        c.require_str("act_fn", "silu", &["silu", "swish"])?;
        c.require_str("resnet_time_scale_shift", "default", &["default"])?;
        c.require_str("time_embedding_type", "positional", &["positional"])?;
        c.require_str("attention_type", "default", &["default"])?;
        c.require_str("mid_block_type", "UNetMidBlock2DCrossAttn", &["UNetMidBlock2DCrossAttn"])?;
        if c.get("mid_block_type").is_none() && c.value.get("mid_block_type").is_some() {
            return Err(c.unsupported("mid_block_type"));
        }
        c.require_str("addition_embed_type", "", &["text_time"])?;

        let block_out_channels = c.usize_list("block_out_channels", &[320, 640, 1280, 1280])?;
        let n = block_out_channels.len();
        let kinds = |key: &str, default: &[&str], allowed: [(&str, BlockKind); 2]| -> Result<Vec<BlockKind>> {
            let names = c.str_list(key, default)?;
            if names.len() != n {
                return Err(c.unsupported(key));
            }
            names
                .iter()
                .map(|name| {
                    allowed.iter().find(|(a, _)| a == name).map(|(_, k)| k.clone()).ok_or_else(|| c.unsupported(key))
                })
                .collect()
        };
        let down_blocks = kinds(
            "down_block_types",
            &["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
            [("DownBlock2D", BlockKind::Plain), ("CrossAttnDownBlock2D", BlockKind::CrossAttn)],
        )?;
        let up_blocks = kinds(
            "up_block_types",
            &["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
            [("UpBlock2D", BlockKind::Plain), ("CrossAttnUpBlock2D", BlockKind::CrossAttn)],
        )?;
        let layers_per_block = c.per_block("layers_per_block", 2, n)?;
        let transformer_layers =
            nested_layers(c, "transformer_layers_per_block", c.get("transformer_layers_per_block"), &layers_per_block)?;
        let reversed_layers: Vec<usize> = layers_per_block.iter().rev().map(|l| l + 1).collect();
        let reverse_transformer_layers = match c.get("reverse_transformer_layers_per_block") {
            None => {
                // reversed(transformer_layers_per_block): each up block uses the per-block value of
                // the mirrored down block, applied to all of its (layers + 1) resnets when it is an int.
                let raw = c.get("transformer_layers_per_block");
                let mut out = Vec::new();
                for (i, &count) in reversed_layers.iter().enumerate() {
                    let j = n - 1 - i;
                    let value = match raw {
                        None => Value::Number("1".into(), 1.0),
                        Some(Value::Array(items)) => items[j].clone(),
                        Some(v) => v.clone(),
                    };
                    out.push(match &value {
                        Value::Array(items) => {
                            // A per-layer list sized for the down block is indexed per up layer.
                            let list: Vec<usize> =
                                items.iter().filter_map(|x| x.as_i64().map(|i| i as usize)).collect();
                            if up_blocks[i] == BlockKind::CrossAttn && list.len() < count {
                                return Err(c.unsupported("reverse_transformer_layers_per_block"));
                            }
                            list
                        }
                        v => vec![v.as_i64().unwrap_or(1) as usize; count],
                    });
                }
                out
            }
            Some(v) => nested_layers(c, "reverse_transformer_layers_per_block", Some(v), &reversed_layers)?,
        };
        let norm_num_groups = match c.value.get("norm_num_groups") {
            None => 32,
            Some(_) => {
                c.opt_usize("norm_num_groups")?.filter(|g| *g > 0).ok_or_else(|| c.unsupported("norm_num_groups"))?
            }
        };
        let heads = c.per_block("attention_head_dim", 8, n)?;
        let cross_attention_dim = c.per_block("cross_attention_dim", 1280, n)?;
        let time_embed_dim = c.opt_usize("time_embedding_dim")?.unwrap_or(block_out_channels[0] * 4);
        let addition_time_embed_dim =
            c.opt_usize("addition_time_embed_dim")?.ok_or_else(|| c.unsupported("addition_time_embed_dim"))?;
        let projection_class_embeddings_input_dim = c
            .opt_usize("projection_class_embeddings_input_dim")?
            .ok_or_else(|| c.unsupported("projection_class_embeddings_input_dim"))?;
        let conv_in_kernel = c.usize_or("conv_in_kernel", 3)?;
        let conv_out_kernel = c.usize_or("conv_out_kernel", 3)?;
        if conv_in_kernel % 2 == 0 || conv_out_kernel % 2 == 0 {
            return Err(c.unsupported("conv_in_kernel"));
        }
        for (i, &ch) in block_out_channels.iter().enumerate() {
            if heads[i] == 0 || ch % heads[i] != 0 {
                return Err(c.unsupported("attention_head_dim"));
            }
            if ch % norm_num_groups != 0 {
                return Err(c.unsupported("norm_num_groups"));
            }
        }
        Ok(UNetConfig {
            in_channels: c.usize_or("in_channels", 4)?,
            out_channels: c.usize_or("out_channels", 4)?,
            center_input_sample: c.bool_or("center_input_sample", false)?,
            flip_sin_to_cos: c.bool_or("flip_sin_to_cos", true)?,
            freq_shift: c.f64_or("freq_shift", 0.0)?,
            down_blocks,
            up_blocks,
            layers_per_block,
            downsample_padding: c.usize_or("downsample_padding", 1)?,
            mid_block_scale_factor: c.f64_or("mid_block_scale_factor", 1.0)?,
            norm_num_groups,
            norm_eps: c.f64_or("norm_eps", 1e-5)?,
            cross_attention_dim,
            transformer_layers,
            reverse_transformer_layers,
            heads,
            use_linear_projection: c.bool_or("use_linear_projection", false)?,
            addition_time_embed_dim,
            projection_class_embeddings_input_dim,
            time_embed_dim,
            conv_in_kernel,
            conv_out_kernel,
            block_out_channels,
        })
    }
}

/// `get_timestep_embedding` (float32), for each value in `timesteps`.
pub fn timestep_embedding(timesteps: &[f32], dim: usize, flip_sin_to_cos: bool, shift: f64) -> Vec<f32> {
    let half = dim / 2;
    let log = -(10000f64.ln()) as f32;
    let divisor = (half as f64 - shift) as f32;
    let freqs: Vec<f32> = (0..half).map(|i| ((log * i as f32) / divisor).exp()).collect();
    let mut out = Vec::with_capacity(timesteps.len() * dim);
    for &t in timesteps {
        let args: Vec<f32> = freqs.iter().map(|f| t * f).collect();
        let sin = args.iter().map(|a| a.sin());
        let cos = args.iter().map(|a| a.cos());
        if flip_sin_to_cos {
            out.extend(cos.chain(sin));
        } else {
            out.extend(sin.chain(cos));
        }
        if dim % 2 == 1 {
            out.push(0.0);
        }
    }
    out
}

struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    fn load(w: &mut Weights, prefix: &str, input: usize, dim: usize) -> Result<TimestepEmbedding> {
        Ok(TimestepEmbedding {
            linear_1: Linear::load(w, &format!("{prefix}.linear_1"), input, dim, true)?,
            linear_2: Linear::load(w, &format!("{prefix}.linear_2"), dim, dim, true)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear_2.forward(&crate::nn::silu(&self.linear_1.forward(x)?)?)
    }
}

pub struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    time_emb_proj: Option<Linear>,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
    output_scale_factor: f64,
}

impl ResnetBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        w: &mut Weights,
        prefix: &str,
        input: usize,
        output: usize,
        temb: Option<usize>,
        groups: usize,
        eps: f64,
        output_scale_factor: f64,
    ) -> Result<ResnetBlock> {
        Ok(ResnetBlock {
            norm1: GroupNorm::load(w, &format!("{prefix}.norm1"), groups, input, eps)?,
            conv1: Conv2d::load(w, &format!("{prefix}.conv1"), input, output, 3, 1, 1, true)?,
            time_emb_proj: match temb {
                Some(t) => Some(Linear::load(w, &format!("{prefix}.time_emb_proj"), t, output, true)?),
                None => None,
            },
            norm2: GroupNorm::load(w, &format!("{prefix}.norm2"), groups, output, eps)?,
            conv2: Conv2d::load(w, &format!("{prefix}.conv2"), output, output, 3, 1, 1, true)?,
            shortcut: if input != output {
                Some(Conv2d::load(w, &format!("{prefix}.conv_shortcut"), input, output, 1, 1, 0, true)?)
            } else {
                None
            },
            output_scale_factor,
        })
    }

    pub fn forward(&self, x: &Tensor, temb: Option<&Tensor>) -> Result<Tensor> {
        let mut h = self.conv1.forward(&crate::nn::silu(&self.norm1.forward(x)?)?)?;
        if let (Some(proj), Some(temb)) = (&self.time_emb_proj, temb) {
            let t = proj.forward(&crate::nn::silu(temb)?)?.unsqueeze(2).and_then(|t| t.unsqueeze(3)).ok_or_failed()?;
            h = h.broadcast_add(&t).ok_or_failed()?;
        }
        let h = self.conv2.forward(&crate::nn::silu(&self.norm2.forward(&h)?)?)?;
        let input = match &self.shortcut {
            Some(conv) => conv.forward(x)?,
            None => x.clone(),
        };
        let out = (input + h).ok_or_failed()?;
        if self.output_scale_factor == 1.0 {
            Ok(out)
        } else {
            crate::nn::div_scalar(&out, self.output_scale_factor as f32)
        }
    }
}

struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
}

impl Attention {
    fn load(
        w: &mut Weights,
        prefix: &str,
        query_dim: usize,
        context_dim: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Attention> {
        let inner = heads * head_dim;
        Ok(Attention {
            to_q: Linear::load(w, &format!("{prefix}.to_q"), query_dim, inner, false)?,
            to_k: Linear::load(w, &format!("{prefix}.to_k"), context_dim, inner, false)?,
            to_v: Linear::load(w, &format!("{prefix}.to_v"), context_dim, inner, false)?,
            to_out: Linear::load(w, &format!("{prefix}.to_out.0"), inner, query_dim, true)?,
            heads,
        })
    }

    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let q = split_heads(&self.to_q.forward(x)?, self.heads)?;
        let k = split_heads(&self.to_k.forward(context)?, self.heads)?;
        let v = split_heads(&self.to_v.forward(context)?, self.heads)?;
        self.to_out.forward(&merge_heads(&attention(&q, &k, &v, None)?)?)
    }
}

struct TransformerBlock {
    norm1: LayerNorm,
    attn1: Attention,
    norm2: LayerNorm,
    attn2: Attention,
    norm3: LayerNorm,
    ff_proj: Linear,
    ff_out: Linear,
}

impl TransformerBlock {
    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let n = self.norm1.forward(x)?;
        let x = (self.attn1.forward(&n, &n)? + x).ok_or_failed()?;
        let n = self.norm2.forward(&x)?;
        let x = (self.attn2.forward(&n, context)? + &x).ok_or_failed()?;
        let n = self.norm3.forward(&x)?;
        let projected = self.ff_proj.forward(&n)?;
        let parts = projected.chunk(2, candle_core::D::Minus1).ok_or_failed()?;
        let gated = (&parts[0] * Activation::Gelu.forward(&parts[1])?).ok_or_failed()?;
        (self.ff_out.forward(&gated)? + &x).ok_or_failed()
    }
}

enum Projection {
    Linear(Linear),
    Conv(Conv2d),
}

struct Transformer2D {
    norm: GroupNorm,
    proj_in: Projection,
    blocks: Vec<TransformerBlock>,
    proj_out: Projection,
}

impl Transformer2D {
    #[allow(clippy::too_many_arguments)]
    fn load(
        w: &mut Weights,
        prefix: &str,
        channels: usize,
        heads: usize,
        layers: usize,
        context_dim: usize,
        groups: usize,
        linear: bool,
    ) -> Result<Transformer2D> {
        let head_dim = channels / heads;
        let inner = heads * head_dim;
        let projection = |w: &mut Weights, name: &str, input: usize, output: usize| -> Result<Projection> {
            Ok(if linear {
                Projection::Linear(Linear::load(w, &format!("{prefix}.{name}"), input, output, true)?)
            } else {
                Projection::Conv(Conv2d::load(w, &format!("{prefix}.{name}"), input, output, 1, 1, 0, true)?)
            })
        };
        let norm = GroupNorm::load(w, &format!("{prefix}.norm"), groups, channels, 1e-6)?;
        let proj_in = projection(w, "proj_in", channels, inner)?;
        let mut blocks = Vec::new();
        for i in 0..layers {
            let b = format!("{prefix}.transformer_blocks.{i}");
            blocks.push(TransformerBlock {
                norm1: LayerNorm::load(w, &format!("{b}.norm1"), inner, 1e-5)?,
                attn1: Attention::load(w, &format!("{b}.attn1"), inner, inner, heads, head_dim)?,
                norm2: LayerNorm::load(w, &format!("{b}.norm2"), inner, 1e-5)?,
                attn2: Attention::load(w, &format!("{b}.attn2"), inner, context_dim, heads, head_dim)?,
                norm3: LayerNorm::load(w, &format!("{b}.norm3"), inner, 1e-5)?,
                ff_proj: Linear::load(w, &format!("{b}.ff.net.0.proj"), inner, inner * 4 * 2, true)?,
                ff_out: Linear::load(w, &format!("{b}.ff.net.2"), inner * 4, inner, true)?,
            });
        }
        let proj_out = projection(w, "proj_out", inner, channels)?;
        Ok(Transformer2D { norm, proj_in, blocks, proj_out })
    }

    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (b, c, h, wd) = x.dims4().ok_or_failed()?;
        let normed = self.norm.forward(x)?;
        let to_tokens = |t: &Tensor| -> Result<Tensor> {
            let inner = t.dims()[1];
            t.permute((0, 2, 3, 1)).and_then(|t| t.reshape((b, h * wd, inner))).ok_or_failed()
        };
        let mut tokens = match &self.proj_in {
            Projection::Linear(l) => l.forward(&to_tokens(&normed)?)?,
            Projection::Conv(conv) => to_tokens(&conv.forward(&normed)?)?,
        };
        for block in &self.blocks {
            tokens = block.forward(&tokens, context)?;
        }
        let inner = tokens.dims()[2];
        let to_map = |t: &Tensor, ch: usize| -> Result<Tensor> {
            t.reshape((b, h, wd, ch)).and_then(|t| t.permute((0, 3, 1, 2))).and_then(|t| t.contiguous()).ok_or_failed()
        };
        let out = match &self.proj_out {
            Projection::Linear(l) => to_map(&l.forward(&tokens)?, c)?,
            Projection::Conv(conv) => conv.forward(&to_map(&tokens, inner)?)?,
        };
        (out + x).ok_or_failed()
    }
}

pub struct Downsample {
    conv: Conv2d,
    asymmetric: bool,
}

impl Downsample {
    pub fn load(w: &mut Weights, prefix: &str, channels: usize, padding: usize) -> Result<Downsample> {
        Ok(Downsample {
            conv: Conv2d::load(w, &format!("{prefix}.conv"), channels, channels, 3, 2, padding, true)?,
            asymmetric: padding == 0,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.asymmetric {
            let padded = x.pad_with_zeros(3, 0, 1).and_then(|t| t.pad_with_zeros(2, 0, 1)).ok_or_failed()?;
            self.conv.forward(&padded)
        } else {
            self.conv.forward(x)
        }
    }
}

pub struct Upsample {
    conv: Conv2d,
}

impl Upsample {
    pub fn load(w: &mut Weights, prefix: &str, channels: usize) -> Result<Upsample> {
        Ok(Upsample { conv: Conv2d::load(w, &format!("{prefix}.conv"), channels, channels, 3, 1, 1, true)? })
    }

    pub fn forward(&self, x: &Tensor, size: Option<(usize, usize)>) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4().ok_or_failed()?;
        let (oh, ow) = size.unwrap_or((h * 2, w * 2));
        self.conv.forward(&upsample_nearest(x, oh, ow)?)
    }
}

struct DownBlock {
    resnets: Vec<ResnetBlock>,
    attentions: Vec<Transformer2D>,
    downsample: Option<Downsample>,
}

struct UpBlock {
    resnets: Vec<ResnetBlock>,
    attentions: Vec<Transformer2D>,
    upsample: Option<Upsample>,
}

pub struct UNet {
    pub config: UNetConfig,
    conv_in: Conv2d,
    time_embedding: TimestepEmbedding,
    add_embedding: TimestepEmbedding,
    down: Vec<DownBlock>,
    mid_resnets: Vec<ResnetBlock>,
    mid_attention: Transformer2D,
    up: Vec<UpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl UNet {
    pub fn load(w: &mut Weights, config: UNetConfig) -> Result<UNet> {
        let c = &config;
        let n = c.block_out_channels.len();
        let ch0 = c.block_out_channels[0];
        let temb = c.time_embed_dim;
        let (groups, eps) = (c.norm_num_groups, c.norm_eps);
        let conv_in =
            Conv2d::load(w, "conv_in", c.in_channels, ch0, c.conv_in_kernel, 1, (c.conv_in_kernel - 1) / 2, true)?;
        let time_embedding = TimestepEmbedding::load(w, "time_embedding", ch0, temb)?;
        let add_embedding = TimestepEmbedding::load(w, "add_embedding", c.projection_class_embeddings_input_dim, temb)?;
        let mut down = Vec::new();
        let mut output = ch0;
        for i in 0..n {
            let input = output;
            output = c.block_out_channels[i];
            let mut resnets = Vec::new();
            let mut attentions = Vec::new();
            for j in 0..c.layers_per_block[i] {
                let p = format!("down_blocks.{i}");
                resnets.push(ResnetBlock::load(
                    w,
                    &format!("{p}.resnets.{j}"),
                    if j == 0 { input } else { output },
                    output,
                    Some(temb),
                    groups,
                    eps,
                    1.0,
                )?);
                if c.down_blocks[i] == BlockKind::CrossAttn {
                    attentions.push(Transformer2D::load(
                        w,
                        &format!("{p}.attentions.{j}"),
                        output,
                        c.heads[i],
                        c.transformer_layers[i][j],
                        c.cross_attention_dim[i],
                        groups,
                        c.use_linear_projection,
                    )?);
                }
            }
            let downsample = if i + 1 < n {
                Some(Downsample::load(w, &format!("down_blocks.{i}.downsamplers.0"), output, c.downsample_padding)?)
            } else {
                None
            };
            down.push(DownBlock { resnets, attentions, downsample });
        }
        let last = c.block_out_channels[n - 1];
        let mid_layers = c.transformer_layers[n - 1][0];
        let mid_resnets = vec![
            ResnetBlock::load(w, "mid_block.resnets.0", last, last, Some(temb), groups, eps, c.mid_block_scale_factor)?,
            ResnetBlock::load(w, "mid_block.resnets.1", last, last, Some(temb), groups, eps, c.mid_block_scale_factor)?,
        ];
        let mid_attention = Transformer2D::load(
            w,
            "mid_block.attentions.0",
            last,
            c.heads[n - 1],
            mid_layers,
            c.cross_attention_dim[n - 1],
            groups,
            c.use_linear_projection,
        )?;
        let reversed: Vec<usize> = c.block_out_channels.iter().rev().copied().collect();
        let mut up = Vec::new();
        let mut output = reversed[0];
        for i in 0..n {
            let prev = output;
            output = reversed[i];
            let input = reversed[(i + 1).min(n - 1)];
            let layers = c.layers_per_block[n - 1 - i] + 1;
            let heads = c.heads[n - 1 - i];
            let context = c.cross_attention_dim[n - 1 - i];
            let mut resnets = Vec::new();
            let mut attentions = Vec::new();
            for j in 0..layers {
                let skip = if j == layers - 1 { input } else { output };
                let resnet_in = if j == 0 { prev } else { output };
                let p = format!("up_blocks.{i}");
                resnets.push(ResnetBlock::load(
                    w,
                    &format!("{p}.resnets.{j}"),
                    resnet_in + skip,
                    output,
                    Some(temb),
                    groups,
                    eps,
                    1.0,
                )?);
                if c.up_blocks[i] == BlockKind::CrossAttn {
                    attentions.push(Transformer2D::load(
                        w,
                        &format!("{p}.attentions.{j}"),
                        output,
                        heads,
                        c.reverse_transformer_layers[i][j],
                        context,
                        groups,
                        c.use_linear_projection,
                    )?);
                }
            }
            let upsample =
                if i + 1 < n { Some(Upsample::load(w, &format!("up_blocks.{i}.upsamplers.0"), output)?) } else { None };
            up.push(UpBlock { resnets, attentions, upsample });
        }
        let conv_norm_out = GroupNorm::load(w, "conv_norm_out", groups, ch0, eps)?;
        let conv_out =
            Conv2d::load(w, "conv_out", ch0, c.out_channels, c.conv_out_kernel, 1, (c.conv_out_kernel - 1) / 2, true)?;
        Ok(UNet {
            config,
            conv_in,
            time_embedding,
            add_embedding,
            down,
            mid_resnets,
            mid_attention,
            up,
            conv_norm_out,
            conv_out,
        })
    }

    /// One denoiser evaluation: `sample` (B, C, h, w), the scalar timestep, the text embeddings
    /// (B, 77, D), the pooled embeddings (B, P) and the time ids (B, 6) as plain values.
    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: f32,
        context: &Tensor,
        pooled: &Tensor,
        time_ids: &[Vec<f32>],
    ) -> Result<Tensor> {
        let c = &self.config;
        let device: &Device = sample.device();
        let dtype = sample.dtype();
        let batch = sample.dims()[0];
        let ch0 = c.block_out_channels[0];
        let t_emb = timestep_embedding(&vec![timestep; batch], ch0, c.flip_sin_to_cos, c.freq_shift);
        let t_emb = Tensor::from_vec(t_emb, (batch, ch0), device).and_then(|t| t.to_dtype(dtype)).ok_or_failed()?;
        let mut emb = self.time_embedding.forward(&t_emb)?;
        let flat: Vec<f32> = time_ids.iter().flatten().copied().collect();
        let per = c.addition_time_embed_dim;
        let time_embeds = timestep_embedding(&flat, per, c.flip_sin_to_cos, c.freq_shift);
        let width = flat.len() / batch * per;
        let time_embeds =
            Tensor::from_vec(time_embeds, (batch, width), device).and_then(|t| t.to_dtype(dtype)).ok_or_failed()?;
        let add = Tensor::cat(&[&pooled.to_dtype(dtype).ok_or_failed()?, &time_embeds], 1).ok_or_failed()?;
        if add.dims()[1] != c.projection_class_embeddings_input_dim {
            return Err(Error::Unsupported("unet.projection_class_embeddings_input_dim".into()));
        }
        emb = (emb + self.add_embedding.forward(&add)?).ok_or_failed()?;

        let mut x = if c.center_input_sample { sample.affine(2.0, -1.0).ok_or_failed()? } else { sample.clone() };
        let (_, _, h, w) = sample.dims4().ok_or_failed()?;
        let factor = 1usize << (c.block_out_channels.len() - 1);
        let forward_upsample_size = h % factor != 0 || w % factor != 0;
        x = self.conv_in.forward(&x)?;
        let mut stack = vec![x.clone()];
        for block in &self.down {
            for (j, resnet) in block.resnets.iter().enumerate() {
                x = resnet.forward(&x, Some(&emb))?;
                if let Some(attn) = block.attentions.get(j) {
                    x = attn.forward(&x, context)?;
                }
                stack.push(x.clone());
            }
            if let Some(down) = &block.downsample {
                x = down.forward(&x)?;
                stack.push(x.clone());
            }
        }
        x = self.mid_resnets[0].forward(&x, Some(&emb))?;
        x = self.mid_attention.forward(&x, context)?;
        x = self.mid_resnets[1].forward(&x, Some(&emb))?;
        for (i, block) in self.up.iter().enumerate() {
            let count = block.resnets.len();
            let mut res: Vec<Tensor> = stack.split_off(stack.len() - count);
            let size = if i + 1 < self.up.len() && forward_upsample_size {
                let last = stack.last().unwrap();
                Some((last.dims()[2], last.dims()[3]))
            } else {
                None
            };
            for (j, resnet) in block.resnets.iter().enumerate() {
                let skip = res.pop().unwrap();
                x = Tensor::cat(&[&x, &skip], 1).ok_or_failed()?;
                x = resnet.forward(&x, Some(&emb))?;
                if let Some(attn) = block.attentions.get(j) {
                    x = attn.forward(&x, context)?;
                }
            }
            if let Some(upsample) = &block.upsample {
                x = upsample.forward(&x, size)?;
            }
        }
        x = crate::nn::silu(&self.conv_norm_out.forward(&x)?)?;
        self.conv_out.forward(&x)
    }
}
