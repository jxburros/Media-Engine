//! Layers on candle-core ops (PyTorch semantics) and the strict weight store they load from.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor, D};

use crate::json;
use crate::Error;

pub type Result<T> = std::result::Result<T, Error>;

pub fn failed(e: candle_core::Error) -> Error {
    // The first line only: candle appends a backtrace when RUST_BACKTRACE is set.
    let text = e.to_string();
    Error::Failed(format!("tensor: {}", text.lines().next().unwrap_or("")))
}

pub trait Fallible<T> {
    fn ok_or_failed(self) -> Result<T>;
}

impl<T> Fallible<T> for candle_core::Result<T> {
    fn ok_or_failed(self) -> Result<T> {
        self.map_err(failed)
    }
}

/// `t ∘ s` for a Python number or CPU 0-d tensor `s`: torch evaluates the op in the dtype's
/// opmath (float32 for half/bfloat16) and rounds the result once to the tensor's dtype.
fn scalar_op(t: &Tensor, value: f32, op: fn(&Tensor, &Tensor) -> candle_core::Result<Tensor>) -> Result<Tensor> {
    let dtype = t.dtype();
    let s = Tensor::new(value, t.device()).ok_or_failed()?;
    if dtype == DType::F32 {
        op(t, &s).ok_or_failed()
    } else if dtype == DType::F64 {
        op(t, &s.to_dtype(DType::F64).ok_or_failed()?).ok_or_failed()
    } else {
        op(&t.to_dtype(DType::F32).ok_or_failed()?, &s).and_then(|r| r.to_dtype(dtype)).ok_or_failed()
    }
}

pub fn mul_scalar(t: &Tensor, value: f32) -> Result<Tensor> {
    scalar_op(t, value, |a, b| a.broadcast_mul(b))
}

pub fn div_scalar(t: &Tensor, value: f32) -> Result<Tensor> {
    scalar_op(t, value, |a, b| a.broadcast_div(b))
}

pub fn add_scalar(t: &Tensor, value: f32) -> Result<Tensor> {
    scalar_op(t, value, |a, b| a.broadcast_add(b))
}

/// On CPU, half and bfloat16 products are computed in float32 and rounded once, as torch's CPU
/// kernels (oneDNN/MKL GEMM and convolution, the flash-attention kernel) accumulate in float32.
pub fn cpu_reduced(t: &Tensor) -> bool {
    t.device().is_cpu() && matches!(t.dtype(), DType::F16 | DType::BF16)
}

/// `a @ b` (with broadcasting of `a` over `b`'s batch when `broadcast`). candle's CPU backend has
/// no bfloat16 matmul; torch's CPU bfloat16 GEMM accumulates in float32 and rounds the result,
/// which is what the float32 product cast back to bfloat16 computes.
pub fn matmul(a: &Tensor, b: &Tensor, broadcast: bool) -> Result<Tensor> {
    let product = |a: &Tensor, b: &Tensor| if broadcast { a.broadcast_matmul(b) } else { a.matmul(b) };
    if a.dtype() == DType::BF16 && a.device().is_cpu() {
        let (a32, b32) = (a.to_dtype(DType::F32).ok_or_failed()?, b.to_dtype(DType::F32).ok_or_failed()?);
        return product(&a32, &b32).and_then(|t| t.to_dtype(DType::BF16)).ok_or_failed();
    }
    product(a, b).ok_or_failed()
}

// ----- weights -------------------------------------------------------------------------------

/// A component's safetensors files, as `OwnedPipelineBackend.weight_files`.
pub fn weight_files(folder: &Path) -> Result<Vec<PathBuf>> {
    let mut indexes = Vec::new();
    let mut files = Vec::new();
    let entries = std::fs::read_dir(folder).map_err(|e| Error::Failed(format!("component folder: {e}")))?;
    for entry in entries {
        let entry = entry.map_err(|e| Error::Failed(format!("component folder: {e}")))?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".safetensors.index.json") {
            indexes.push(entry.path());
        } else if name.ends_with(".safetensors") {
            files.push(entry.path());
        }
    }
    indexes.sort();
    files.sort();
    if indexes.len() == 1 {
        let text = std::fs::read_to_string(&indexes[0]).map_err(|e| Error::Failed(format!("weight index: {e}")))?;
        let index = json::parse(&text).map_err(|e| Error::Failed(format!("weight index: {e}")))?;
        let map = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| Error::Failed("weight index has no weight_map".into()))?;
        let mut shards: Vec<String> = map.iter().filter_map(|(_, v)| v.as_str().map(str::to_string)).collect();
        shards.sort();
        shards.dedup();
        return Ok(shards.into_iter().map(|s| folder.join(s)).collect());
    }
    if files.len() == 1 {
        return Ok(files);
    }
    let standard: Vec<PathBuf> = files
        .into_iter()
        .filter(|f| {
            let name = f.file_name().unwrap().to_string_lossy();
            name == "diffusion_pytorch_model.safetensors" || name == "model.safetensors"
        })
        .collect();
    if standard.len() == 1 {
        return Ok(standard);
    }
    Err(Error::Failed("component weights are ambiguous".into()))
}

/// A component's state dict, cast to the load dtype; every tensor must be consumed exactly once
/// with the expected shape (`load_state_dict(strict=True)`).
pub struct Weights {
    tensors: HashMap<String, Tensor>,
    used: HashSet<String>,
    component: String,
}

impl Weights {
    pub fn load(folder: &Path, component: &str, dtype: DType, device: &Device) -> Result<Weights> {
        let files = weight_files(folder)?;
        let mut tensors = HashMap::new();
        for file in files {
            // Later files win, as `state.update(load_file(path))`.
            let loaded = candle_core::safetensors::load(&file, &Device::Cpu)
                .map_err(|e| Error::Failed(format!("{component} weights: {e}")))?;
            for (name, tensor) in loaded {
                let tensor = if tensor.dtype().is_float() { tensor.to_dtype(dtype).ok_or_failed()? } else { tensor };
                tensors.insert(name, tensor.to_device(device).ok_or_failed()?);
            }
        }
        Ok(Weights { tensors, used: HashSet::new(), component: component.to_string() })
    }

    /// Cast every floating tensor (the VAE's float16 → float32 upcast).
    pub fn cast(mut self, dtype: DType) -> Result<Weights> {
        for tensor in self.tensors.values_mut() {
            if tensor.dtype().is_float() {
                *tensor = tensor.to_dtype(dtype).ok_or_failed()?;
            }
        }
        Ok(self)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        self.tensors.get_mut(name)
    }

    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Failed(format!("{} weights are missing {name}", self.component)))?;
        if tensor.dims() != shape {
            return Err(Error::Failed(format!(
                "{} weight {name} has shape {:?}, expected {:?}",
                self.component,
                tensor.dims(),
                shape
            )));
        }
        if !self.used.insert(name.to_string()) {
            return Err(Error::Failed(format!("{} weight {name} used twice", self.component)));
        }
        Ok(tensor.clone())
    }

    /// Every stored tensor was consumed (no unexpected keys).
    pub fn finish(self) -> Result<()> {
        let mut extra: Vec<&String> = self.tensors.keys().filter(|k| !self.used.contains(*k)).collect();
        extra.sort();
        if let Some(first) = extra.first() {
            return Err(Error::Failed(format!(
                "{} weights have {} unexpected tensors (first: {first})",
                self.component,
                extra.len()
            )));
        }
        Ok(())
    }
}

// ----- layers --------------------------------------------------------------------------------

pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    /// The parameters' dtype (the weights themselves are float32 on CPU for reduced precision).
    dtype: DType,
}

/// The tensor to compute with: float32 on CPU for reduced precision, else itself.
fn compute_copy(t: Tensor) -> Result<Tensor> {
    if cpu_reduced(&t) {
        t.to_dtype(DType::F32).ok_or_failed()
    } else {
        Ok(t)
    }
}

impl Linear {
    pub fn load(w: &mut Weights, prefix: &str, input: usize, output: usize, bias: bool) -> Result<Linear> {
        let weight = w.take(&format!("{prefix}.weight"), &[output, input])?;
        let dtype = weight.dtype();
        Ok(Linear {
            weight: compute_copy(weight)?,
            bias: if bias { Some(compute_copy(w.take(&format!("{prefix}.bias"), &[output])?)?) } else { None },
            dtype,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let input = *dims.last().unwrap();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let flat = x.reshape((rows, input)).and_then(|t| t.to_dtype(self.weight.dtype())).ok_or_failed()?;
        let mut y = matmul(&flat, &self.weight.t().ok_or_failed()?, false)?;
        if let Some(b) = &self.bias {
            y = y.broadcast_add(b).ok_or_failed()?;
        }
        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = self.weight.dims()[0];
        y.reshape(out_dims).and_then(|t| t.to_dtype(self.dtype)).ok_or_failed()
    }
}

pub struct Conv2d {
    dtype: DType,
    weight: Tensor,
    /// The kernel as one (C_out, C_in) matrix per tap, for the CPU path.
    taps: Vec<Tensor>,
    kernel: usize,
    bias: Option<Tensor>,
    padding: usize,
    stride: usize,
}

impl Conv2d {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        w: &mut Weights,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Result<Conv2d> {
        let weight = w.take(&format!("{prefix}.weight"), &[output, input, kernel, kernel])?;
        let dtype = weight.dtype();
        let weight = compute_copy(weight)?;
        let mut taps = Vec::new();
        if weight.device().is_cpu() {
            for ky in 0..kernel {
                for kx in 0..kernel {
                    taps.push(
                        weight
                            .narrow(2, ky, 1)
                            .and_then(|t| t.narrow(3, kx, 1))
                            .and_then(|t| t.reshape((output, input)))
                            .and_then(|t| t.contiguous())
                            .ok_or_failed()?,
                    );
                }
            }
        }
        Ok(Conv2d {
            dtype,
            weight,
            taps,
            kernel,
            bias: if bias { Some(compute_copy(w.take(&format!("{prefix}.bias"), &[output])?)?) } else { None },
            padding,
            stride,
        })
    }

    /// `nn.Conv2d` (zero padding). On CPU the convolution is computed here as one matrix product
    /// per kernel tap, accumulated in the input dtype (candle-core 0.9.2's tiled CPU kernel gives
    /// wrong results for some shapes); other devices use candle's convolution.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(self.weight.dtype()).ok_or_failed()?;
        let mut y = if self.taps.is_empty() {
            x.conv2d(&self.weight, self.padding, self.stride, 1, 1).ok_or_failed()?
        } else {
            self.forward_taps(&x)?
        };
        if let Some(b) = &self.bias {
            let c = b.dims()[0];
            y = y.broadcast_add(&b.reshape((1, c, 1, 1)).ok_or_failed()?).ok_or_failed()?;
        }
        y.to_dtype(self.dtype).ok_or_failed()
    }

    fn forward_taps(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4().ok_or_failed()?;
        let (k, s, p) = (self.kernel, self.stride, self.padding);
        let out_c = self.weight.dims()[0];
        if h + 2 * p < k || w + 2 * p < k {
            return Err(Error::Failed("convolution input smaller than its kernel".into()));
        }
        let oh = (h + 2 * p - k) / s + 1;
        let ow = (w + 2 * p - k) / s + 1;
        let padded = if p > 0 {
            x.pad_with_zeros(2, p, p).and_then(|t| t.pad_with_zeros(3, p, p)).ok_or_failed()?
        } else {
            x.clone()
        };
        let rows = if s > 1 {
            Some(
                Tensor::new((0..oh as u32).map(|i| i * s as u32).collect::<Vec<_>>().as_slice(), x.device())
                    .ok_or_failed()?,
            )
        } else {
            None
        };
        let cols = if s > 1 {
            Some(
                Tensor::new((0..ow as u32).map(|i| i * s as u32).collect::<Vec<_>>().as_slice(), x.device())
                    .ok_or_failed()?,
            )
        } else {
            None
        };
        let mut acc: Option<Tensor> = None;
        for ky in 0..k {
            for kx in 0..k {
                let mut slice = padded
                    .narrow(2, ky, (oh - 1) * s + 1)
                    .and_then(|t| t.narrow(3, kx, (ow - 1) * s + 1))
                    .ok_or_failed()?;
                if let (Some(r), Some(cl)) = (&rows, &cols) {
                    slice = slice
                        .contiguous()
                        .and_then(|t| t.index_select(r, 2))
                        .and_then(|t| t.index_select(cl, 3))
                        .ok_or_failed()?;
                }
                let columns = slice.contiguous().and_then(|t| t.reshape((b, c, oh * ow))).ok_or_failed()?;
                let product = matmul(&self.taps[ky * k + kx], &columns, true)?;
                acc = Some(match acc {
                    None => product,
                    Some(a) => (a + product).ok_or_failed()?,
                });
            }
        }
        acc.unwrap().reshape((b, out_c, oh, ow)).ok_or_failed()
    }
}

pub struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    groups: usize,
    eps: f64,
}

impl GroupNorm {
    pub fn load(w: &mut Weights, prefix: &str, groups: usize, channels: usize, eps: f64) -> Result<GroupNorm> {
        if groups == 0 || !channels.is_multiple_of(groups) {
            return Err(Error::Unsupported("norm_num_groups".into()));
        }
        Ok(GroupNorm {
            weight: w.take(&format!("{prefix}.weight"), &[channels])?,
            bias: w.take(&format!("{prefix}.bias"), &[channels])?,
            groups,
            eps,
        })
    }

    /// `nn.GroupNorm` over (B, C, …); statistics in float32 (as torch does for reduced types).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let (b, c) = (dims[0], dims[1]);
        let dtype = x.dtype();
        let grouped = x.to_dtype(DType::F32).and_then(|t| t.reshape((b, self.groups, ()))).ok_or_failed()?;
        let mean = grouped.mean_keepdim(2).ok_or_failed()?;
        let centered = grouped.broadcast_sub(&mean).ok_or_failed()?;
        let var = centered.sqr().and_then(|t| t.mean_keepdim(2)).ok_or_failed()?;
        let normed = centered
            .broadcast_div(&var.affine(1.0, self.eps).and_then(|t| t.sqrt()).ok_or_failed()?)
            .and_then(|t| t.reshape(dims.as_slice()))
            .ok_or_failed()?;
        let mut shape = vec![1usize; dims.len()];
        shape[1] = c;
        let weight = self.weight.to_dtype(DType::F32).and_then(|t| t.reshape(shape.as_slice())).ok_or_failed()?;
        let bias = self.bias.to_dtype(DType::F32).and_then(|t| t.reshape(shape.as_slice())).ok_or_failed()?;
        normed
            .broadcast_mul(&weight)
            .and_then(|t| t.broadcast_add(&bias))
            .and_then(|t| t.to_dtype(dtype))
            .ok_or_failed()
    }
}

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    pub fn load(w: &mut Weights, prefix: &str, dim: usize, eps: f64) -> Result<LayerNorm> {
        Ok(LayerNorm {
            weight: w.take(&format!("{prefix}.weight"), &[dim])?,
            bias: w.take(&format!("{prefix}.bias"), &[dim])?,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x32 = x.to_dtype(DType::F32).ok_or_failed()?;
        let mean = x32.mean_keepdim(D::Minus1).ok_or_failed()?;
        let centered = x32.broadcast_sub(&mean).ok_or_failed()?;
        let var = centered.sqr().and_then(|t| t.mean_keepdim(D::Minus1)).ok_or_failed()?;
        let normed =
            centered.broadcast_div(&var.affine(1.0, self.eps).and_then(|t| t.sqrt()).ok_or_failed()?).ok_or_failed()?;
        let weight = self.weight.to_dtype(DType::F32).ok_or_failed()?;
        let bias = self.bias.to_dtype(DType::F32).ok_or_failed()?;
        normed
            .broadcast_mul(&weight)
            .and_then(|t| t.broadcast_add(&bias))
            .and_then(|t| t.to_dtype(dtype))
            .ok_or_failed()
    }
}

/// Softmax over the last dimension (computed in float32).
pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32).ok_or_failed()?;
    let max = x.max_keepdim(D::Minus1).ok_or_failed()?;
    let e = x.broadcast_sub(&max).and_then(|t| t.exp()).ok_or_failed()?;
    let sum = e.sum_keepdim(D::Minus1).ok_or_failed()?;
    e.broadcast_div(&sum).and_then(|t| t.to_dtype(dtype)).ok_or_failed()
}

/// Scaled dot-product attention over (B, H, L, Dh) tensors with an optional additive mask
/// broadcastable to (B, H, Lq, Lk); queries are processed in chunks so the score matrix stays
/// bounded (rows of a softmax are independent, so chunking does not change the result).
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    if cpu_reduced(q) {
        // Scores and softmax in float32; the probabilities are rounded to the input dtype before
        // the second product, and the output once more (torch's CPU flash-attention kernel).
        let dtype = q.dtype();
        let f = |t: &Tensor| t.to_dtype(DType::F32).ok_or_failed();
        let mask = match mask {
            Some(m) => Some(f(m)?),
            None => None,
        };
        return attention_f32(&f(q)?, &f(k)?, &f(v)?, mask.as_ref(), Some(dtype))?.to_dtype(dtype).ok_or_failed();
    }
    attention_f32(q, k, v, mask, None)
}

/// Score elements computed at once (256 MiB of float32).
const ATTENTION_BUDGET: usize = 1 << 26;

fn attention_f32(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    probs_dtype: Option<DType>,
) -> Result<Tensor> {
    attention_chunked(q, k, v, mask, probs_dtype, ATTENTION_BUDGET)
}

fn attention_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    probs_dtype: Option<DType>,
    budget: usize,
) -> Result<Tensor> {
    let (b, h, lq, dh) = q.dims4().ok_or_failed()?;
    let lk = k.dims()[2];
    let scale = 1.0 / (dh as f64).sqrt();
    let kt = k.transpose(2, 3).and_then(|t| t.contiguous()).ok_or_failed()?;
    let rows = (budget / (b * h * lk).max(1)).clamp(1, lq);
    let mut outputs = Vec::new();
    let mut start = 0;
    while start < lq {
        let n = rows.min(lq - start);
        let qc = q.narrow(2, start, n).and_then(|t| t.contiguous()).ok_or_failed()?;
        let mut scores = mul_scalar(&matmul(&qc, &kt, false)?, scale as f32)?;
        if let Some(m) = mask {
            let mc = if m.dims()[2] == 1 { m.clone() } else { m.narrow(2, start, n).ok_or_failed()? };
            scores = scores.broadcast_add(&mc).ok_or_failed()?;
        }
        let mut probs = softmax_last(&scores)?;
        if let Some(dtype) = probs_dtype {
            probs = probs.to_dtype(dtype).and_then(|t| t.to_dtype(DType::F32)).ok_or_failed()?;
        }
        outputs.push(matmul(&probs, v, false)?);
        start += n;
    }
    if outputs.len() == 1 {
        Ok(outputs.pop().unwrap())
    } else {
        Tensor::cat(&outputs, 2).ok_or_failed()
    }
}

/// (B, L, H·Dh) → (B, H, L, Dh).
pub fn split_heads(x: &Tensor, heads: usize) -> Result<Tensor> {
    let (b, l, c) = x.dims3().ok_or_failed()?;
    x.reshape((b, l, heads, c / heads)).and_then(|t| t.transpose(1, 2)).and_then(|t| t.contiguous()).ok_or_failed()
}

/// (B, H, L, Dh) → (B, L, H·Dh).
pub fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let (b, h, l, d) = x.dims4().ok_or_failed()?;
    x.transpose(1, 2).and_then(|t| t.contiguous()).and_then(|t| t.reshape((b, l, h * d))).ok_or_failed()
}

/// `F.interpolate(x, size=(out_h, out_w), mode="nearest")` (or `scale_factor=2`).
pub fn upsample_nearest(x: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4().ok_or_failed()?;
    let index = |input: usize, output: usize| -> Vec<u32> {
        (0..output)
            .map(|o| {
                if output == 2 * input {
                    (o >> 1) as u32
                } else {
                    let scale = input as f32 / output as f32;
                    (((o as f32) * scale).floor() as usize).min(input - 1) as u32
                }
            })
            .collect()
    };
    let rows = Tensor::new(index(h, out_h).as_slice(), x.device()).ok_or_failed()?;
    let cols = Tensor::new(index(w, out_w).as_slice(), x.device()).ok_or_failed()?;
    x.contiguous().and_then(|t| t.index_select(&rows, 2)).and_then(|t| t.index_select(&cols, 3)).ok_or_failed()
}

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Silu,
    Gelu,
    GeluTanh,
    QuickGelu,
}

/// `F.silu`.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    Activation::Silu.forward(x)
}

impl Activation {
    /// Evaluated in float32 for half/bfloat16 on CPU and rounded once, as torch's CPU kernels do
    /// (candle's reduced-precision CPU kernels round every intermediate).
    pub fn forward(self, x: &Tensor) -> Result<Tensor> {
        if cpu_reduced(x) {
            let dtype = x.dtype();
            return self.forward_native(&x.to_dtype(DType::F32).ok_or_failed()?)?.to_dtype(dtype).ok_or_failed();
        }
        self.forward_native(x)
    }

    fn forward_native(self, x: &Tensor) -> Result<Tensor> {
        match self {
            Activation::Silu => x.silu().ok_or_failed(),
            Activation::Gelu => x.gelu_erf().ok_or_failed(),
            Activation::GeluTanh => x.gelu().ok_or_failed(),
            Activation::QuickGelu => {
                let gate =
                    x.affine(-1.702, 0.0).and_then(|t| t.exp()).and_then(|t| t.affine(1.0, 1.0)).ok_or_failed()?;
                x.broadcast_div(&gate).ok_or_failed()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_attention_equals_unchunked() {
        let dev = Device::Cpu;
        let q = Tensor::randn(0f32, 1.0, (2, 3, 37, 8), &dev).unwrap();
        let k = Tensor::randn(0f32, 1.0, (2, 3, 29, 8), &dev).unwrap();
        let v = Tensor::randn(0f32, 1.0, (2, 3, 29, 8), &dev).unwrap();
        let mask = Tensor::randn(0f32, 1.0, (1, 1, 37, 29), &dev).unwrap();
        let whole = attention_chunked(&q, &k, &v, Some(&mask), None, usize::MAX).unwrap();
        let chunked = attention_chunked(&q, &k, &v, Some(&mask), None, 2 * 3 * 29 * 5).unwrap();
        let diff = (whole - chunked).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn convolution_matches_direct_sum() {
        // Strided, padded and unpadded cases against a float64 direct convolution.
        let dev = Device::Cpu;
        for (c_in, c_out, size, k, stride, pad) in [(3, 5, 9, 3, 1, 1), (4, 6, 10, 3, 2, 1), (4, 4, 11, 3, 2, 0), (6, 2, 7, 1, 1, 0)] {
            let x = Tensor::randn(0f32, 1.0, (2, c_in, size, size), &dev).unwrap();
            let w = Tensor::randn(0f32, 1.0, (c_out, c_in, k, k), &dev).unwrap();
            let mut store = Weights { tensors: [("c.weight".to_string(), w.clone())].into(), used: HashSet::new(), component: "t".into() };
            let conv = Conv2d::load(&mut store, "c", c_in, c_out, k, stride, pad, false).unwrap();
            let y = conv.forward(&x).unwrap();
            let (xv, wv) = (x.flatten_all().unwrap().to_vec1::<f32>().unwrap(), w.flatten_all().unwrap().to_vec1::<f32>().unwrap());
            let (_, _, oh, ow) = y.dims4().unwrap();
            assert_eq!(oh, (size + 2 * pad - k) / stride + 1);
            let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for n in 0..2 {
                for o in 0..c_out {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let mut acc = 0f64;
                            for c in 0..c_in {
                                for ky in 0..k {
                                    for kx in 0..k {
                                        let iy = (oy * stride + ky) as isize - pad as isize;
                                        let ix = (ox * stride + kx) as isize - pad as isize;
                                        if iy < 0 || ix < 0 || iy >= size as isize || ix >= size as isize {
                                            continue;
                                        }
                                        acc += xv[((n * c_in + c) * size + iy as usize) * size + ix as usize] as f64
                                            * wv[((o * c_in + c) * k + ky) * k + kx] as f64;
                                    }
                                }
                            }
                            let got = yv[((n * c_out + o) * oh + oy) * ow + ox] as f64;
                            assert!((got - acc).abs() < 1e-4, "{got} vs {acc}");
                        }
                    }
                }
            }
        }
    }
}
