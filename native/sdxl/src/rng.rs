//! PyTorch's CPU generator: `torch.Generator().manual_seed(seed)` followed by `torch.randn`.
//!
//! The engine is `at::mt19937` (seeded with the low 32 bits, like `CPUGeneratorImpl`), uniform
//! floats take the low 24 bits of one 32-bit draw, and `torch.randn` of 16 or more contiguous
//! elements fills every element with a uniform first and then applies Box–Muller in blocks of 16
//! (`normal_fill`, aten/src/ATen/native/cpu/DistributionTemplates.h): element j of a block pairs
//! with element j + 8, and when the count is not a multiple of 16 the last 16 elements are drawn
//! again and recomputed. Smaller tensors use the serial double-precision path with the
//! generator's cached second sample. Which float32 kernel torch runs depends on the host's ATen
//! CPU capability: AVX2/AVX512 hosts run the avx_mathfun polynomials (compiled with FMA
//! contraction), the default build (no AVX2, or non-x86) the C library's `logf`/`cosf`/`sinf`;
//! both are reproduced bit for bit and selected as ATen selects them (honoring
//! `ATEN_CPU_CAPABILITY`).

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UMASK: u32 = 0x8000_0000;
const LMASK: u32 = 0x7fff_ffff;

/// The element type `torch.randn` is asked for; values are returned as f32 holding exactly the
/// value torch stores in that dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RandnDtype {
    Float32,
    Float16,
    BFloat16,
}

/// The float32 Box–Muller kernel ATen dispatches to on this host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalKernel {
    /// `normal_fill` with libm `logf`/`cosf`/`sinf` (ATen's default build).
    Scalar,
    /// `normal_fill_AVX2` with avx_mathfun's `log256_ps`/`sincos256_ps`.
    Avx2,
}

impl NormalKernel {
    /// ATen's choice: `ATEN_CPU_CAPABILITY` if set, else the host's best capability. Both the
    /// AVX2 and the AVX512 builds of torch 2.6 produce the AVX2 kernel's bits (verified against
    /// torch on an AVX512 host and with `ATEN_CPU_CAPABILITY=avx2`); only the default build
    /// (no AVX2/FMA, or non-x86) runs the libm kernel.
    pub fn detect() -> NormalKernel {
        if let Ok(value) = std::env::var("ATEN_CPU_CAPABILITY") {
            match value.to_ascii_lowercase().as_str() {
                "avx2" | "avx512" => return NormalKernel::Avx2,
                "default" => return NormalKernel::Scalar,
                _ => {}
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return NormalKernel::Avx2;
            }
        }
        NormalKernel::Scalar
    }
}

#[derive(Clone)]
pub struct TorchGenerator {
    state: [u32; N],
    left: usize,
    next: usize,
    next_double_normal: Option<f64>,
    kernel: NormalKernel,
}

impl TorchGenerator {
    /// `torch.Generator().manual_seed(seed)` with the host's kernel selection.
    pub fn new(seed: u64) -> TorchGenerator {
        TorchGenerator::with_kernel(seed, NormalKernel::detect())
    }

    pub fn with_kernel(seed: u64, kernel: NormalKernel) -> TorchGenerator {
        let mut state = [0u32; N];
        state[0] = (seed & 0xffff_ffff) as u32;
        for j in 1..N {
            state[j] = 1_812_433_253u32.wrapping_mul(state[j - 1] ^ (state[j - 1] >> 30)).wrapping_add(j as u32);
        }
        TorchGenerator { state, left: 1, next: 0, next_double_normal: None, kernel }
    }

    fn twist(u: u32, v: u32) -> u32 {
        (((u & UMASK) | (v & LMASK)) >> 1) ^ if v & 1 != 0 { MATRIX_A } else { 0 }
    }

    fn next_state(&mut self) {
        let s = &mut self.state;
        for p in 0..N - M {
            s[p] = s[p + M] ^ Self::twist(s[p], s[p + 1]);
        }
        for p in N - M..N - 1 {
            s[p] = s[p + M - N] ^ Self::twist(s[p], s[p + 1]);
        }
        s[N - 1] = s[M - 1] ^ Self::twist(s[N - 1], s[0]);
        self.left = N;
        self.next = 0;
    }

    /// One 32-bit draw (`CPUGeneratorImpl::random`).
    pub fn random(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.next_state();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `CPUGeneratorImpl::random64`: two draws, the first in the high word.
    pub fn random64(&mut self) -> u64 {
        let hi = self.random() as u64;
        let lo = self.random() as u64;
        (hi << 32) | lo
    }

    fn uniform_bits(&mut self, digits: u32) -> f32 {
        let mask = (1u32 << digits) - 1;
        ((self.random() & mask) as f32) * (1.0 / (1u64 << digits) as f32)
    }

    fn uniform_f64(&mut self) -> f64 {
        let mask = (1u64 << 53) - 1;
        ((self.random64() & mask) as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// `at::normal_distribution<double>` with the generator's cached second sample.
    fn normal_f64(&mut self) -> f64 {
        if let Some(cached) = self.next_double_normal.take() {
            return cached;
        }
        let u1 = self.uniform_f64();
        let u2 = self.uniform_f64();
        let r = (-2.0f64 * (-u2).ln_1p()).sqrt();
        let theta = 2.0f64 * std::f64::consts::PI * u1;
        self.next_double_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    /// `torch.randn(numel, dtype=dtype, generator=self)` for a contiguous tensor.
    pub fn randn(&mut self, numel: usize, dtype: RandnDtype) -> Vec<f32> {
        if numel < 16 {
            return (0..numel)
                .map(|_| {
                    let v = self.normal_f64();
                    match dtype {
                        RandnDtype::Float32 => v as f32,
                        RandnDtype::Float16 => round_f16(v as f32),
                        RandnDtype::BFloat16 => round_bf16(v as f32),
                    }
                })
                .collect();
        }
        let digits = match dtype {
            RandnDtype::Float32 => 24,
            RandnDtype::Float16 => 11,
            RandnDtype::BFloat16 => 8,
        };
        let mut data: Vec<f32> = (0..numel).map(|_| self.uniform_bits(digits)).collect();
        let block = |gen: &Self, chunk: &mut [f32]| match dtype {
            RandnDtype::Float32 => match gen.kernel {
                NormalKernel::Scalar => fill16_scalar(chunk),
                NormalKernel::Avx2 => fill16_avx2(chunk),
            },
            RandnDtype::Float16 => fill16_reduced(chunk, round_f16),
            RandnDtype::BFloat16 => fill16_reduced(chunk, round_bf16),
        };
        let mut i = 0;
        while i + 16 <= numel {
            block(self, &mut data[i..i + 16]);
            i += 16;
        }
        if !numel.is_multiple_of(16) {
            let start = numel - 16;
            for value in data[start..].iter_mut() {
                *value = self.uniform_bits(digits);
            }
            block(self, &mut data[start..]);
        }
        data
    }
}

/// `normal_fill_16<float>` (mean 0, std 1).
fn fill16_scalar(data: &mut [f32]) {
    for j in 0..8 {
        let u1 = 1.0f32 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0f32 * u1.ln()).sqrt();
        let theta = (2.0f32 as f64 * std::f64::consts::PI * u2 as f64) as f32;
        data[j] = radius * theta.cos() * 1.0 + 0.0;
        data[j + 8] = radius * theta.sin() * 1.0 + 0.0;
    }
}

/// `normal_fill_16<c10::Half>` / `<c10::BFloat16>`: c10's math overloads return the reduced
/// type, so every intermediate (1 − u, log, −2·log, sqrt, θ, cos/sin, the product) rounds.
fn fill16_reduced(data: &mut [f32], round: fn(f32) -> f32) {
    for j in 0..8 {
        let u1 = round(1.0f32 - data[j]);
        let u2 = data[j + 8];
        let radius = round(round(-2.0f32 * round(u1.ln())).sqrt());
        let theta = round((2.0f32 as f64 * std::f64::consts::PI * u2 as f64) as f32);
        data[j] = round(radius * round(theta.cos()));
        data[j + 8] = round(radius * round(theta.sin()));
    }
}

/// `normal_fill_16_AVX2`, lane by lane.
fn fill16_avx2(data: &mut [f32]) {
    let two_pi = (2.0f32 as f64 * std::f64::consts::PI) as f32;
    for j in 0..8 {
        let u1 = 1.0f32 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0f32 * avx_log(u1)).sqrt();
        let theta = two_pi * u2;
        let (s, c) = avx_sincos(theta);
        data[j] = (radius * c).mul_add(1.0, 0.0);
        data[j + 8] = (radius * s).mul_add(1.0, 0.0);
    }
}

/// avx_mathfun `log256_ps` for one lane (its Cephes constants, verbatim).
#[allow(clippy::excessive_precision, clippy::approx_constant)]
fn avx_log(x: f32) -> f32 {
    const SQRTHF: f32 = 0.707_106_781_186_547_524;
    const P: [f32; 9] = [
        7.037_683_629_2E-2,
        -1.151_461_031_0E-1,
        1.167_699_874_0E-1,
        -1.242_014_084_6E-1,
        1.424_932_278_7E-1,
        -1.666_805_766_5E-1,
        2.000_071_476_5E-1,
        -2.499_999_399_3E-1,
        3.333_333_117_4E-1,
    ];
    const Q1: f32 = -2.121_944_40e-4;
    const Q2: f32 = 0.693_359_375;
    let invalid = x <= 0.0;
    let x0 = x.max(f32::from_bits(0x0080_0000));
    let bits = x0.to_bits();
    let imm0 = (bits >> 23) as i32 - 0x7f;
    let mut x = f32::from_bits((bits & !0x7f80_0000) | 0.5f32.to_bits());
    let mut e = imm0 as f32 + 1.0;
    let mask = x < SQRTHF;
    let tmp = if mask { x } else { 0.0 };
    x -= 1.0;
    e -= if mask { 1.0 } else { 0.0 };
    x += tmp;
    // The AVX2 kernel is compiled with FMA contraction: every multiply feeding an add fuses.
    let z = x * x;
    let mut y = P[0];
    for p in &P[1..] {
        y = y.mul_add(x, *p);
    }
    y *= x;
    y = y.mul_add(z, e * Q1);
    y = (-z).mul_add(0.5, y);
    x += y;
    x = e.mul_add(Q2, x);
    if invalid {
        f32::NAN
    } else {
        x
    }
}

/// avx_mathfun `sincos256_ps` for one lane: (sin, cos).
#[allow(clippy::excessive_precision, clippy::approx_constant)]
fn avx_sincos(x: f32) -> (f32, f32) {
    const DP1: f32 = -0.785_156_25;
    const DP2: f32 = -2.418_756_484_985_351_562_5e-4;
    const DP3: f32 = -3.774_894_977_445_941_08e-8;
    const SINCOF: [f32; 3] = [-1.951_529_589_1E-4, 8.332_160_873_6E-3, -1.666_665_461_1E-1];
    const COSCOF: [f32; 3] = [2.443_315_711_809_948E-005, -1.388_731_625_493_765E-003, 4.166_664_568_298_827E-002];
    const FOPI: f32 = 1.273_239_544_735_16;
    let sign_sin = x.to_bits() & 0x8000_0000;
    let mut x = f32::from_bits(x.to_bits() & 0x7fff_ffff);
    let y = x * FOPI;
    let mut imm2 = (y as i32).wrapping_add(1) & !1;
    let y = imm2 as f32;
    let imm4 = imm2;
    let swap_sin = ((imm2 & 4) as u32) << 29;
    imm2 &= 2;
    let poly_mask = imm2 == 0;
    x = y.mul_add(DP1, x);
    x = y.mul_add(DP2, x);
    x = y.mul_add(DP3, x);
    let sign_cos = ((!(imm4.wrapping_sub(2)) & 4) as u32) << 29;
    let sign_sin = sign_sin ^ swap_sin;
    let z = x * x;
    let mut c = COSCOF[0];
    c = c.mul_add(z, COSCOF[1]);
    c = c.mul_add(z, COSCOF[2]);
    c *= z;
    c = c.mul_add(z, -(z * 0.5));
    c += 1.0;
    let mut s = SINCOF[0];
    s = s.mul_add(z, SINCOF[1]);
    s = s.mul_add(z, SINCOF[2]);
    s *= z;
    s = s.mul_add(x, x);
    let (sin, cos) = if poly_mask { (0.0 + s, c + 0.0) } else { (c + 0.0, 0.0 + s) };
    (f32::from_bits(sin.to_bits() ^ sign_sin), f32::from_bits(cos.to_bits() ^ sign_cos))
}

/// f32 → IEEE half (round to nearest even) → f32.
pub fn round_f16(x: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(x))
}

/// f32 → bfloat16 (round to nearest even, as c10::BFloat16) → f32.
pub fn round_bf16(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits(0x7fc0_0000);
    }
    let bits = x.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits((bits.wrapping_add(rounding)) & 0xffff_0000)
}

pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let rounded = if rem > halfway || (rem == halfway && half & 1 == 1) { half + 1 } else { half };
        return sign | rounded as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let rounded = if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) { half + 1 } else { half };
    sign | rounded as u16
}

pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_reference_stream() {
        // std::mt19937 with the default seed 5489: the 10000th output is 4123659995.
        let mut g = TorchGenerator::with_kernel(5489, NormalKernel::Scalar);
        let mut last = 0;
        for _ in 0..10000 {
            last = g.random();
        }
        assert_eq!(last, 4_123_659_995);
    }

    fn fnv(values: &[f32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in values {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h
    }

    /// (case, seed, draws as (numel, dtype)) and the FNV-1a hash of each draw's f32 bits, recorded
    /// from torch 2.6.0 (`torch.Generator().manual_seed(seed)`; one `torch.randn` per draw, in
    /// order, from the same generator) on the AVX2/AVX512 build and on `ATEN_CPU_CAPABILITY=default`.
    #[allow(clippy::type_complexity)]
    const CASES: &[(&str, u64, &[(usize, RandnDtype)], &[u64], &[u64])] = &[
        (
            "latent",
            7,
            &[(4096, RandnDtype::Float32), (4096, RandnDtype::Float32)],
            &[0x3857b45e32574168, 0xaa97c6e4e63db55c],
            &[0x2719c3ab86ee4f9d, 0xb84257a8679fdded],
        ),
        (
            "tail",
            123,
            &[(20, RandnDtype::Float32), (37, RandnDtype::Float32)],
            &[0x6d4856a80fe2bc42, 0xd85d23062298fb71],
            &[0xe11db79e3beec6eb, 0xa3a79982a3fd8d69],
        ),
        (
            "serial",
            42,
            &[(5, RandnDtype::Float32), (3, RandnDtype::Float32), (17, RandnDtype::Float32), (2, RandnDtype::Float32)],
            &[0x8181828dbc48cfb7, 0xcaad871574dc5d75, 0x8dcc6a39e02dc365, 0xde7892ba03e06e24],
            &[0x8181828dbc48cfb7, 0xcaad871574dc5d75, 0x47ea152dd9f4e584, 0xde7892ba03e06e24],
        ),
        (
            "half",
            7,
            &[(256, RandnDtype::Float16), (5, RandnDtype::Float16)],
            &[0x9003d1c02b5c5725, 0xff406392db6cc94f],
            &[0x9003d1c02b5c5725, 0xff406392db6cc94f],
        ),
        (
            "bfloat",
            7,
            &[(256, RandnDtype::BFloat16), (5, RandnDtype::BFloat16)],
            &[0xadaf3109d6adb725, 0x69823e40c85ce94f],
            &[0xadaf3109d6adb725, 0x69823e40c85ce94f],
        ),
        (
            "seed above 2^32",
            (1u64 << 40) + 5,
            &[(16, RandnDtype::Float32)],
            &[0xeb2ad8e702860c71],
            &[0xf210d49708939658],
        ),
    ];

    #[test]
    fn randn_matches_torch_bit_for_bit() {
        for (name, seed, draws, avx2, default) in CASES {
            for (kernel, expected) in [(NormalKernel::Avx2, avx2), (NormalKernel::Scalar, default)] {
                let mut g = TorchGenerator::with_kernel(*seed, kernel);
                for (i, (numel, dtype)) in draws.iter().enumerate() {
                    let values = g.randn(*numel, *dtype);
                    assert_eq!(values.len(), *numel);
                    assert_eq!(fnv(&values), expected[i], "{name} draw {i} ({kernel:?})");
                }
            }
        }
        let mut g = TorchGenerator::with_kernel(7, NormalKernel::Avx2);
        let first = g.randn(4096, RandnDtype::Float32);
        assert_eq!(&first[..3], &[-0.820_134_5f32, 0.395_631_16, 0.898_908_5]);
    }

    #[test]
    fn half_rounding() {
        assert_eq!(round_f16(1.0 + 1.0 / 4096.0), 1.0);
        assert_eq!(round_f16(1.0 + 3.0 / 2048.0), 1.0 + 2.0 / 1024.0);
        assert_eq!(round_f16(65520.0), f32::INFINITY);
        assert_eq!(round_f16(1e-7), 1.0 / 16777216.0 * 2.0);
        assert_eq!(round_bf16(1.0 + 1.0 / 256.0), 1.0);
    }
}
