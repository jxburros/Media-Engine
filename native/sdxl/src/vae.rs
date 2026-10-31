//! diffusers' `AutoencoderKL` (DownEncoderBlock2D / UpDecoderBlock2D, mid-block attention,
//! quant / post-quant convolutions).

use candle_core::Tensor;

use crate::config::Config;
use crate::nn::{attention, merge_heads, split_heads, Conv2d, Fallible, GroupNorm, Linear, Result, Weights};
use crate::unet::{Downsample, ResnetBlock, Upsample};

#[derive(Clone, Debug)]
pub struct VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub scaling_factor: f64,
    pub force_upcast: bool,
    pub use_quant_conv: bool,
    pub use_post_quant_conv: bool,
    pub mid_block_add_attention: bool,
}

impl VaeConfig {
    pub fn parse(c: &Config) -> Result<VaeConfig> {
        let block_out_channels = c.usize_list("block_out_channels", &[64])?;
        let n = block_out_channels.len();
        for (key, name) in [("down_block_types", "DownEncoderBlock2D"), ("up_block_types", "UpDecoderBlock2D")] {
            let types = c.str_list(key, &[name])?;
            if types.len() != n || types.iter().any(|t| t != name) {
                return Err(c.unsupported(key));
            }
        }
        c.require_str("act_fn", "silu", &["silu", "swish"])?;
        // The owned pipeline applies only `scaling_factor`; a shift or per-channel statistics would
        // change the latent space's meaning.
        c.require_null("shift_factor")?;
        c.require_null("latents_mean")?;
        c.require_null("latents_std")?;
        let norm_num_groups = c.usize_or("norm_num_groups", 32)?;
        for &ch in &block_out_channels {
            if norm_num_groups == 0 || ch % norm_num_groups != 0 {
                return Err(c.unsupported("norm_num_groups"));
            }
        }
        Ok(VaeConfig {
            in_channels: c.usize_or("in_channels", 3)?,
            out_channels: c.usize_or("out_channels", 3)?,
            layers_per_block: c.usize_or("layers_per_block", 1)?,
            latent_channels: c.usize_or("latent_channels", 4)?,
            norm_num_groups,
            scaling_factor: c.f64_or("scaling_factor", 0.18215)?,
            force_upcast: c.bool_or("force_upcast", true)?,
            use_quant_conv: c.bool_or("use_quant_conv", true)?,
            use_post_quant_conv: c.bool_or("use_post_quant_conv", true)?,
            mid_block_add_attention: c.bool_or("mid_block_add_attention", true)?,
            block_out_channels,
        })
    }

    pub fn scale_factor(&self) -> usize {
        1 << (self.block_out_channels.len() - 1)
    }
}

struct SelfAttention {
    group_norm: GroupNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
}

impl SelfAttention {
    fn load(w: &mut Weights, prefix: &str, channels: usize, groups: usize) -> Result<SelfAttention> {
        Ok(SelfAttention {
            group_norm: GroupNorm::load(w, &format!("{prefix}.group_norm"), groups, channels, 1e-6)?,
            to_q: Linear::load(w, &format!("{prefix}.to_q"), channels, channels, true)?,
            to_k: Linear::load(w, &format!("{prefix}.to_k"), channels, channels, true)?,
            to_v: Linear::load(w, &format!("{prefix}.to_v"), channels, channels, true)?,
            to_out: Linear::load(w, &format!("{prefix}.to_out.0"), channels, channels, true)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4().ok_or_failed()?;
        let tokens = self
            .group_norm
            .forward(x)?
            .reshape((b, c, h * w))
            .and_then(|t| t.transpose(1, 2))
            .and_then(|t| t.contiguous())
            .ok_or_failed()?;
        let q = split_heads(&self.to_q.forward(&tokens)?, 1)?;
        let k = split_heads(&self.to_k.forward(&tokens)?, 1)?;
        let v = split_heads(&self.to_v.forward(&tokens)?, 1)?;
        let out = self.to_out.forward(&merge_heads(&attention(&q, &k, &v, None)?)?)?;
        let out = out.transpose(1, 2).and_then(|t| t.reshape((b, c, h, w))).ok_or_failed()?;
        (out + x).ok_or_failed()
    }
}

struct MidBlock {
    resnets: Vec<ResnetBlock>,
    attention: Option<SelfAttention>,
}

impl MidBlock {
    fn load(w: &mut Weights, prefix: &str, channels: usize, groups: usize, attention: bool) -> Result<MidBlock> {
        let resnet = |w: &mut Weights, i: usize| {
            ResnetBlock::load(w, &format!("{prefix}.resnets.{i}"), channels, channels, None, groups, 1e-6, 1.0)
        };
        let first = resnet(w, 0)?;
        let attention = if attention {
            Some(SelfAttention::load(w, &format!("{prefix}.attentions.0"), channels, groups)?)
        } else {
            None
        };
        let second = resnet(w, 1)?;
        Ok(MidBlock { resnets: vec![first, second], attention })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.resnets[0].forward(x, None)?;
        if let Some(attn) = &self.attention {
            x = attn.forward(&x)?;
        }
        self.resnets[1].forward(&x, None)
    }
}

pub struct Vae {
    pub config: VaeConfig,
    enc_conv_in: Conv2d,
    enc_down: Vec<(Vec<ResnetBlock>, Option<Downsample>)>,
    enc_mid: MidBlock,
    enc_norm_out: GroupNorm,
    enc_conv_out: Conv2d,
    quant_conv: Option<Conv2d>,
    post_quant_conv: Option<Conv2d>,
    dec_conv_in: Conv2d,
    dec_mid: MidBlock,
    dec_up: Vec<(Vec<ResnetBlock>, Option<Upsample>)>,
    dec_norm_out: GroupNorm,
    dec_conv_out: Conv2d,
}

impl Vae {
    pub fn load(w: &mut Weights, config: VaeConfig) -> Result<Vae> {
        let c = &config;
        let n = c.block_out_channels.len();
        let g = c.norm_num_groups;
        let ch = &c.block_out_channels;
        let last = ch[n - 1];
        let latent = c.latent_channels;
        let enc_conv_in = Conv2d::load(w, "encoder.conv_in", c.in_channels, ch[0], 3, 1, 1, true)?;
        let mut enc_down = Vec::new();
        let mut output = ch[0];
        for i in 0..n {
            let input = output;
            output = ch[i];
            let mut resnets = Vec::new();
            for j in 0..c.layers_per_block {
                resnets.push(ResnetBlock::load(
                    w,
                    &format!("encoder.down_blocks.{i}.resnets.{j}"),
                    if j == 0 { input } else { output },
                    output,
                    None,
                    g,
                    1e-6,
                    1.0,
                )?);
            }
            let down = if i + 1 < n {
                Some(Downsample::load(w, &format!("encoder.down_blocks.{i}.downsamplers.0"), output, 0)?)
            } else {
                None
            };
            enc_down.push((resnets, down));
        }
        let enc_mid = MidBlock::load(w, "encoder.mid_block", last, g, c.mid_block_add_attention)?;
        let enc_norm_out = GroupNorm::load(w, "encoder.conv_norm_out", g, last, 1e-6)?;
        let enc_conv_out = Conv2d::load(w, "encoder.conv_out", last, 2 * latent, 3, 1, 1, true)?;
        let quant_conv = if c.use_quant_conv {
            Some(Conv2d::load(w, "quant_conv", 2 * latent, 2 * latent, 1, 1, 0, true)?)
        } else {
            None
        };
        let post_quant_conv = if c.use_post_quant_conv {
            Some(Conv2d::load(w, "post_quant_conv", latent, latent, 1, 1, 0, true)?)
        } else {
            None
        };
        let dec_conv_in = Conv2d::load(w, "decoder.conv_in", latent, last, 3, 1, 1, true)?;
        let dec_mid = MidBlock::load(w, "decoder.mid_block", last, g, c.mid_block_add_attention)?;
        let reversed: Vec<usize> = ch.iter().rev().copied().collect();
        let mut dec_up = Vec::new();
        let mut output = reversed[0];
        for i in 0..n {
            let prev = output;
            output = reversed[i];
            let mut resnets = Vec::new();
            for j in 0..c.layers_per_block + 1 {
                resnets.push(ResnetBlock::load(
                    w,
                    &format!("decoder.up_blocks.{i}.resnets.{j}"),
                    if j == 0 { prev } else { output },
                    output,
                    None,
                    g,
                    1e-6,
                    1.0,
                )?);
            }
            let up = if i + 1 < n {
                Some(Upsample::load(w, &format!("decoder.up_blocks.{i}.upsamplers.0"), output)?)
            } else {
                None
            };
            dec_up.push((resnets, up));
        }
        let dec_norm_out = GroupNorm::load(w, "decoder.conv_norm_out", g, ch[0], 1e-6)?;
        let dec_conv_out = Conv2d::load(w, "decoder.conv_out", ch[0], c.out_channels, 3, 1, 1, true)?;
        Ok(Vae {
            config,
            enc_conv_in,
            enc_down,
            enc_mid,
            enc_norm_out,
            enc_conv_out,
            quant_conv,
            post_quant_conv,
            dec_conv_in,
            dec_mid,
            dec_up,
            dec_norm_out,
            dec_conv_out,
        })
    }

    /// The posterior's (mean, std) for an image in [-1, 1].
    pub fn encode(&self, image: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut x = self.enc_conv_in.forward(image)?;
        for (resnets, down) in &self.enc_down {
            for r in resnets {
                x = r.forward(&x, None)?;
            }
            if let Some(d) = down {
                x = d.forward(&x)?;
            }
        }
        x = self.enc_mid.forward(&x)?;
        x = self.enc_conv_out.forward(&crate::nn::silu(&self.enc_norm_out.forward(&x)?)?)?;
        if let Some(q) = &self.quant_conv {
            x = q.forward(&x)?;
        }
        let parts = x.chunk(2, 1).ok_or_failed()?;
        let logvar = parts[1].clamp(-30f32, 20f32).ok_or_failed()?;
        let std = crate::nn::mul_scalar(&logvar, 0.5)?.exp().ok_or_failed()?;
        Ok((parts[0].contiguous().ok_or_failed()?, std))
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let mut x = match &self.post_quant_conv {
            Some(p) => p.forward(latents)?,
            None => latents.clone(),
        };
        x = self.dec_conv_in.forward(&x)?;
        x = self.dec_mid.forward(&x)?;
        for (resnets, up) in &self.dec_up {
            for r in resnets {
                x = r.forward(&x, None)?;
            }
            if let Some(u) = up {
                x = u.forward(&x, None)?;
            }
        }
        self.dec_conv_out.forward(&crate::nn::silu(&self.dec_norm_out.forward(&x)?)?)
    }
}
