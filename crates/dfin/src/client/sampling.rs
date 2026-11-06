//! Token sampling over a raw logits vector.
//!
//! Order of operations follows llama.cpp's default chain: top-k, top-p, min-p
//! (the last two judged on the temperature-1 distribution), then temperature,
//! softmax and a weighted draw. `temperature <= 0` is greedy argmax.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    /// `0` (or negative) disables top-k.
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 1.0,
            top_k: 40,
            top_p: 1.0,
            min_p: 0.0,
            seed: None,
        }
    }
}

pub struct Sampler {
    params: SamplingParams,
    rng: StdRng,
}

/// Index of the largest logit (first one on ties). Panics on an empty slice.
pub fn argmax(logits: &[f32]) -> usize {
    assert!(!logits.is_empty(), "argmax over empty logits");
    let mut best = 0usize;
    let mut best_v = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_v {
            best = i;
            best_v = v;
        }
    }
    best
}

/// Numerically stable softmax over `xs`, in place.
fn softmax(xs: &mut [f32]) {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 && sum.is_finite() {
        for x in xs.iter_mut() {
            *x /= sum;
        }
    }
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let rng = match params.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };
        Sampler { params, rng }
    }

    pub fn sample(&mut self, logits: &[f32]) -> usize {
        let p = &self.params;
        if p.temperature <= 0.0 || !p.temperature.is_finite() {
            return argmax(logits);
        }
        // Candidates sorted by logit, descending.
        let mut cand: Vec<(usize, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| !v.is_nan())
            .collect();
        if cand.is_empty() {
            return argmax(logits);
        }
        let by_desc = |a: &(usize, f32), b: &(usize, f32)| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        };
        if p.top_k > 0 && (p.top_k as usize) < cand.len() {
            let k = p.top_k as usize;
            cand.select_nth_unstable_by(k - 1, by_desc);
            cand.truncate(k);
        }
        cand.sort_by(by_desc);

        // top-p and min-p on the untempered distribution.
        if p.top_p < 1.0 || p.min_p > 0.0 {
            let mut probs: Vec<f32> = cand.iter().map(|c| c.1).collect();
            softmax(&mut probs);
            let mut keep = cand.len();
            if p.top_p < 1.0 {
                let mut cum = 0.0f32;
                for (i, pr) in probs.iter().enumerate() {
                    cum += pr;
                    if cum >= p.top_p {
                        keep = keep.min(i + 1);
                        break;
                    }
                }
            }
            if p.min_p > 0.0 {
                let threshold = probs[0] * p.min_p;
                let n = probs
                    .iter()
                    .take_while(|&&pr| pr >= threshold)
                    .count()
                    .max(1);
                keep = keep.min(n);
            }
            cand.truncate(keep.max(1));
        }
        if cand.len() == 1 {
            return cand[0].0;
        }

        let mut probs: Vec<f32> = cand.iter().map(|c| c.1 / p.temperature).collect();
        softmax(&mut probs);
        let total: f32 = probs.iter().sum();
        if total.is_nan() || total <= 0.0 || !total.is_finite() {
            return cand[0].0;
        }
        let mut r = self.rng.random_range(0.0..total);
        for (i, pr) in probs.iter().enumerate() {
            if r < *pr {
                return cand[i].0;
            }
            r -= pr;
        }
        cand[cand.len() - 1].0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits() -> Vec<f32> {
        vec![0.1, 2.5, -1.0, 2.4, 0.0, 1.9, -3.0, 2.49]
    }

    #[test]
    fn temperature_zero_is_argmax() {
        let l = logits();
        let mut s = Sampler::new(SamplingParams {
            temperature: 0.0,
            ..Default::default()
        });
        for _ in 0..10 {
            assert_eq!(s.sample(&l), 1);
        }
    }

    #[test]
    fn top_k_one_is_argmax() {
        let l = logits();
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.5,
            top_k: 1,
            seed: Some(1),
            ..Default::default()
        });
        for _ in 0..50 {
            assert_eq!(s.sample(&l), argmax(&l));
        }
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let l = logits();
        let p = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            seed: Some(42),
            ..Default::default()
        };
        let a: Vec<usize> = {
            let mut s = Sampler::new(p.clone());
            (0..32).map(|_| s.sample(&l)).collect()
        };
        let b: Vec<usize> = {
            let mut s = Sampler::new(p);
            (0..32).map(|_| s.sample(&l)).collect()
        };
        assert_eq!(a, b);
        assert!(a.iter().all(|&i| i < l.len()));
    }

    #[test]
    fn min_p_filters_low_tokens() {
        let l = vec![10.0, 0.0, 0.0, 0.0];
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: 0,
            min_p: 0.1,
            seed: Some(3),
            ..Default::default()
        });
        for _ in 0..50 {
            assert_eq!(s.sample(&l), 0);
        }
    }

    #[test]
    fn top_p_small_keeps_best() {
        let l = logits();
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.01,
            seed: Some(9),
            ..Default::default()
        });
        for _ in 0..50 {
            assert_eq!(s.sample(&l), 1);
        }
    }
}
