//! diffusers' `EulerDiscreteScheduler` (s_churn = 0) for the configurations the SDXL family uses:
//! `linspace` / `leading` / `trailing` spacing, linear interpolation, `epsilon` or `v_prediction`.
//! The schedule arithmetic follows torch/numpy's float32/float64 steps.

use crate::config::Config;
use crate::Error;

type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prediction {
    Epsilon,
    VPrediction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Spacing {
    Linspace,
    Leading,
    Trailing,
}

#[derive(Clone, Debug)]
pub struct EulerConfig {
    train_steps: usize,
    alphas_cumprod: Vec<f32>,
    spacing: Spacing,
    steps_offset: i64,
    final_sigma_min: bool,
    pub prediction: Prediction,
}

/// `torch.linspace(start, end, steps, dtype=float32)` on CPU (float32 step; the kernel's
/// `start + step * i` / `end - step * (steps - i - 1)` halves, compiled to fused multiply-adds).
pub fn linspace_f32(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    let half = steps / 2;
    (0..steps)
        .map(|i| if i < half { step.mul_add(i as f32, start) } else { (-step).mul_add((steps - i - 1) as f32, end) })
        .collect()
}

impl EulerConfig {
    pub fn parse(c: &Config) -> Result<EulerConfig> {
        let train_steps = c.usize_or("num_train_timesteps", 1000)?;
        if train_steps < 2 {
            return Err(c.unsupported("num_train_timesteps"));
        }
        let beta_start = c.f64_or("beta_start", 0.0001)?;
        let beta_end = c.f64_or("beta_end", 0.02)?;
        let betas: Vec<f32> = if let Some(trained) = c.get("trained_betas") {
            let list = trained.as_array().ok_or_else(|| c.unsupported("trained_betas"))?;
            if list.len() != train_steps {
                return Err(c.unsupported("trained_betas"));
            }
            list.iter()
                .map(|v| v.as_f64().map(|f| f as f32).ok_or_else(|| c.unsupported("trained_betas")))
                .collect::<Result<_>>()?
        } else {
            match c.str_or("beta_schedule", "linear")?.as_str() {
                "linear" => linspace_f32(beta_start as f32, beta_end as f32, train_steps),
                "scaled_linear" => linspace_f32(beta_start.sqrt() as f32, beta_end.sqrt() as f32, train_steps)
                    .into_iter()
                    .map(|b| b * b)
                    .collect(),
                "squaredcos_cap_v2" => {
                    let alpha_bar = |t: f64| ((t + 0.008) / 1.008 * std::f64::consts::PI / 2.0).cos().powi(2);
                    (0..train_steps)
                        .map(|i| {
                            let t1 = i as f64 / train_steps as f64;
                            let t2 = (i + 1) as f64 / train_steps as f64;
                            (1.0 - alpha_bar(t2) / alpha_bar(t1)).min(0.999) as f32
                        })
                        .collect()
                }
                _ => return Err(c.unsupported("beta_schedule")),
            }
        };
        c.require_bool("rescale_betas_zero_snr", false)?;
        c.require_bool("use_karras_sigmas", false)?;
        c.require_bool("use_exponential_sigmas", false)?;
        c.require_bool("use_beta_sigmas", false)?;
        c.require_str("interpolation_type", "linear", &["linear"])?;
        let prediction = match c.str_or("prediction_type", "epsilon")?.as_str() {
            "epsilon" => Prediction::Epsilon,
            "v_prediction" => Prediction::VPrediction,
            _ => return Err(c.unsupported("prediction_type")),
        };
        let timestep_type = c.require_str("timestep_type", "discrete", &["discrete", "continuous"])?;
        if timestep_type == "continuous" && prediction == Prediction::VPrediction {
            return Err(c.unsupported("timestep_type"));
        }
        let spacing = match c.str_or("timestep_spacing", "linspace")?.as_str() {
            "linspace" => Spacing::Linspace,
            "leading" => Spacing::Leading,
            "trailing" => Spacing::Trailing,
            _ => return Err(c.unsupported("timestep_spacing")),
        };
        let final_sigma_min = match c.str_or("final_sigmas_type", "zero")?.as_str() {
            "zero" => false,
            "sigma_min" => true,
            _ => return Err(c.unsupported("final_sigmas_type")),
        };
        // alphas_cumprod = torch.cumprod(1 - betas): float32 values, double accumulator.
        let mut acc = 1.0f64;
        let alphas_cumprod = betas
            .iter()
            .map(|b| {
                acc *= (1.0f32 - b) as f64;
                acc as f32
            })
            .collect();
        Ok(EulerConfig {
            train_steps,
            alphas_cumprod,
            spacing,
            steps_offset: c.i64_or("steps_offset", 0)?,
            final_sigma_min,
            prediction,
        })
    }

    /// `set_timesteps(steps)`.
    pub fn schedule(&self, steps: usize) -> Result<Schedule> {
        if steps == 0 {
            return Err(Error::InvalidInput("steps must be positive"));
        }
        let t = self.train_steps;
        let timesteps: Vec<f32> = match self.spacing {
            Spacing::Linspace => {
                let stop = (t - 1) as f64;
                let mut v: Vec<f32> = if steps == 1 {
                    vec![0.0]
                } else {
                    let step = stop / (steps - 1) as f64;
                    (0..steps).map(|i| if i == steps - 1 { stop as f32 } else { (i as f64 * step) as f32 }).collect()
                };
                v.reverse();
                v
            }
            Spacing::Leading => {
                let ratio = t / steps;
                (0..steps).rev().map(|i| (i * ratio) as f32 + self.steps_offset as f32).collect()
            }
            Spacing::Trailing => {
                // np.arange(t, 0, -t / steps).round() − 1, with numpy's arange fill.
                let step = -(t as f64 / steps as f64);
                let start = t as f64;
                let len = ((0.0 - start) / step).ceil().max(0.0) as usize;
                let delta = (start + step) - start;
                (0..len).map(|i| ((start + i as f64 * delta).round_ties_even() as f32) - 1.0).collect()
            }
        };
        let full: Vec<f32> = self.alphas_cumprod.iter().map(|a| ((1.0f32 - a) / a).sqrt()).collect();
        let mut sigmas: Vec<f32> = timesteps.iter().map(|&x| interp(x as f64, &full) as f32).collect();
        let last = if self.final_sigma_min {
            let a = self.alphas_cumprod[0];
            ((1.0f32 - a) / a).sqrt()
        } else {
            0.0
        };
        sigmas.push(last);
        let max = sigmas.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let init_noise_sigma = match self.spacing {
            Spacing::Linspace | Spacing::Trailing => max,
            Spacing::Leading => (max * max + 1.0).sqrt(),
        };
        Ok(Schedule { timesteps, sigmas, init_noise_sigma, prediction: self.prediction })
    }
}

/// `np.interp(x, np.arange(len(fp)), fp)` with float64 arithmetic.
fn interp(x: f64, fp: &[f32]) -> f64 {
    let n = fp.len();
    if x.is_nan() {
        return x;
    }
    if x <= 0.0 {
        return fp[0] as f64;
    }
    if x >= (n - 1) as f64 {
        return fp[n - 1] as f64;
    }
    let j = x.floor() as usize;
    if x == j as f64 {
        return fp[j] as f64;
    }
    let slope = (fp[j + 1] as f64 - fp[j] as f64) / 1.0;
    slope * (x - j as f64) + fp[j] as f64
}

#[derive(Clone, Debug)]
pub struct Schedule {
    pub timesteps: Vec<f32>,
    /// One more than the timesteps (the final sigma).
    pub sigmas: Vec<f32>,
    pub init_noise_sigma: f32,
    pub prediction: Prediction,
}

impl Schedule {
    /// `index_for_timestep`: the second match when the timestep repeats, else the first.
    pub fn index_for_timestep(&self, t: f32) -> usize {
        let matches: Vec<usize> = self.timesteps.iter().enumerate().filter(|(_, &v)| v == t).map(|(i, _)| i).collect();
        if matches.len() > 1 {
            matches[1]
        } else {
            matches.first().copied().unwrap_or(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{self, Value};

    fn config_from(value: &Value) -> Config {
        Config { component: "scheduler".into(), value: value.clone() }
    }

    /// Schedules recorded from diffusers 0.32.2 (tests/make_scheduler_fixture.py). Timesteps are
    /// exact; sigmas are float32 values within one ulp (torch's float32 square root on CPU goes
    /// through MKL, which is not always correctly rounded).
    #[test]
    fn schedules_match_diffusers() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/euler_schedules.json");
        let rows = json::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut worst = 0u32;
        for row in rows.as_array().unwrap() {
            let config = EulerConfig::parse(&config_from(row.get("config").unwrap())).unwrap();
            let steps = row.get("steps").unwrap().as_i64().unwrap() as usize;
            let schedule = config.schedule(steps).unwrap();
            let floats = |key: &str| -> Vec<f32> {
                row.get(key).unwrap().as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect()
            };
            assert_eq!(schedule.timesteps, floats("timesteps"), "{steps} steps of {:?}", row.get("config"));
            let want = floats("sigmas");
            assert_eq!(schedule.sigmas.len(), want.len());
            for (got, want) in schedule
                .sigmas
                .iter()
                .zip(&want)
                .chain([(&schedule.init_noise_sigma, &(row.get("init_noise_sigma").unwrap().as_f64().unwrap() as f32))])
            {
                let ulps = (got.to_bits() as i64 - want.to_bits() as i64).unsigned_abs() as u32;
                worst = worst.max(ulps);
                assert!(ulps <= 1, "{steps} steps: sigma {got} vs {want}");
            }
        }
        assert!(worst <= 1);
    }

    #[test]
    fn unsupported_fields_are_named() {
        let bad = json::parse(r#"{"beta_schedule": "scaled_linear", "use_karras_sigmas": true}"#).unwrap();
        match EulerConfig::parse(&config_from(&bad)) {
            Err(Error::Unsupported(field)) => assert_eq!(field, "scheduler.use_karras_sigmas"),
            other => panic!("{:?}", other.map(|_| ())),
        }
        let bad = json::parse(r#"{"timestep_spacing": "karras"}"#).unwrap();
        assert!(
            matches!(EulerConfig::parse(&config_from(&bad)), Err(Error::Unsupported(f)) if f == "scheduler.timestep_spacing")
        );
    }
}
