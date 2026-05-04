use std::ops::Range;

use crate::inference::argmax_first;
use crate::model::PITCH_BINS;

// kernel(d) = max(12 - |d|, 0) is zero outside ±11 bins, so banded forward
// and backtrack are exact, not an approximation.
const BAND: usize = 11;

// emission mixture: with probability OBSERVATION_TRUST the per-frame argmax
// is the true state; the remaining mass is uniform over all bins. matches
// the reference crepe Python.
const OBSERVATION_TRUST: f64 = 0.1;

pub(crate) fn most_likely_path(salience: &[Vec<f32>]) -> Vec<usize> {
    let observations: Vec<usize> = salience.iter().map(|row| argmax_first(row)).collect();
    let log_trans = transition_log_probs();
    let emit = Emissions::new();
    let log_prob = forward(&observations, &log_trans, emit);
    backtrack(&log_prob, &log_trans)
}

#[derive(Clone, Copy)]
struct Emissions {
    on_match: f64,
    on_miss: f64,
}

impl Emissions {
    fn new() -> Self {
        let p_other = (1.0 - OBSERVATION_TRUST) / PITCH_BINS as f64;
        Self {
            on_match: (OBSERVATION_TRUST + p_other).ln(),
            on_miss: p_other.ln(),
        }
    }

    fn log_prob(self, state: usize, observation: usize) -> f64 {
        if state == observation {
            self.on_match
        } else {
            self.on_miss
        }
    }
}

fn band(center: usize) -> Range<usize> {
    center.saturating_sub(BAND)..(center + BAND + 1).min(PITCH_BINS)
}

fn transition_log_probs() -> Vec<f64> {
    let mut log_trans = vec![f64::NEG_INFINITY; PITCH_BINS * PITCH_BINS];
    for j in 0..PITCH_BINS {
        let row_sum: f64 = band(j).map(|k| 12.0 - (j as f64 - k as f64).abs()).sum();
        let log_row_sum = row_sum.ln();
        for k in band(j) {
            let weight = 12.0 - (j as f64 - k as f64).abs();
            log_trans[j * PITCH_BINS + k] = weight.ln() - log_row_sum;
        }
    }
    log_trans
}

fn forward(observations: &[usize], log_trans: &[f64], emit: Emissions) -> Vec<f64> {
    let n_frames = observations.len();
    let mut log_prob = vec![0.0_f64; n_frames * PITCH_BINS];
    let log_uniform_prior = -(PITCH_BINS as f64).ln();

    for (k, slot) in log_prob[..PITCH_BINS].iter_mut().enumerate() {
        *slot = log_uniform_prior + emit.log_prob(k, observations[0]);
    }

    for t in 1..n_frames {
        let (prev_slice, curr_slice) = log_prob.split_at_mut(t * PITCH_BINS);
        let prev = &prev_slice[(t - 1) * PITCH_BINS..t * PITCH_BINS];
        let curr = &mut curr_slice[..PITCH_BINS];
        let obs_t = observations[t];
        for k in 0..PITCH_BINS {
            let mut best = f64::NEG_INFINITY;
            for j in band(k) {
                let s = prev[j] + log_trans[j * PITCH_BINS + k];
                if s > best {
                    best = s;
                }
            }
            curr[k] = best + emit.log_prob(k, obs_t);
        }
    }
    log_prob
}

fn backtrack(log_prob: &[f64], log_trans: &[f64]) -> Vec<usize> {
    let n_frames = log_prob.len() / PITCH_BINS;
    let mut path = vec![0usize; n_frames];

    // final frame uses numpy first-index argmax; earlier frames use hmmlearn
    // last-index. preserved for parity with the python reference.
    let last = &log_prob[(n_frames - 1) * PITCH_BINS..];
    path[n_frames - 1] = argmax_first(last);

    for t in (0..n_frames - 1).rev() {
        let next = path[t + 1];
        let row = &log_prob[t * PITCH_BINS..(t + 1) * PITCH_BINS];
        let mut best = f64::NEG_INFINITY;
        let mut best_idx = band(next).start;
        for j in band(next) {
            let s = row[j] + log_trans[j * PITCH_BINS + next];
            if s >= best {
                best = s;
                best_idx = j;
            }
        }
        path[t] = best_idx;
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logsumexp(xs: &[f64]) -> f64 {
        let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        max + xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
    }

    fn salience_dominant_at(bin: usize, n_frames: usize) -> Vec<Vec<f32>> {
        let p_other = 0.1_f32 / (PITCH_BINS - 1) as f32;
        (0..n_frames)
            .map(|_| {
                let mut row = vec![p_other; PITCH_BINS];
                row[bin] = 0.9;
                row
            })
            .collect()
    }

    #[test]
    fn band_clips_at_edges() {
        assert_eq!(band(0), 0..(BAND + 1));
        assert_eq!(band(PITCH_BINS - 1), (PITCH_BINS - 1 - BAND)..PITCH_BINS);
        let mid = band(180);
        assert_eq!(mid.end - mid.start, 2 * BAND + 1);
    }

    #[test]
    fn transition_rows_sum_to_one() {
        let log_trans = transition_log_probs();
        for j in 0..PITCH_BINS {
            let row = &log_trans[j * PITCH_BINS..(j + 1) * PITCH_BINS];
            let lse = logsumexp(row);
            assert!(lse.abs() < 1e-12, "row {j} logsumexp = {lse}");
        }
    }

    #[test]
    fn transition_is_banded() {
        let log_trans = transition_log_probs();
        for j in 0..PITCH_BINS {
            let row = &log_trans[j * PITCH_BINS..(j + 1) * PITCH_BINS];
            for (k, &v) in row.iter().enumerate() {
                if band(j).contains(&k) {
                    assert!(v.is_finite(), "expected finite at ({j}, {k}), got {v}");
                } else {
                    assert_eq!(v, f64::NEG_INFINITY, "expected -inf at ({j}, {k})");
                }
            }
            let diag = row[j];
            for (k, &v) in row.iter().enumerate() {
                if k != j {
                    assert!(diag >= v, "diag {diag} < row[{k}] {v} for j={j}");
                }
            }
        }
    }

    #[test]
    fn emission_match_beats_miss() {
        let emit = Emissions::new();
        assert!(emit.on_match > emit.on_miss);
        let p_other = (1.0 - OBSERVATION_TRUST) / PITCH_BINS as f64;
        let expected_diff = ((OBSERVATION_TRUST + p_other) / p_other).ln();
        assert!((emit.on_match - emit.on_miss - expected_diff).abs() < 1e-12);
        assert_eq!(emit.log_prob(0, 0), emit.on_match);
        assert_eq!(emit.log_prob(0, 1), emit.on_miss);
    }

    #[test]
    fn path_follows_dominant_bin() {
        let salience = salience_dominant_at(100, 8);
        let path = most_likely_path(&salience);
        assert_eq!(path, vec![100; 8]);
    }

    #[test]
    fn path_smooths_single_frame_outlier() {
        // 10 frames all dominant at bin 100, except frame 5 which spikes at
        // bin 200. the transition kernel only spans ±11 bins, so jumping
        // 100 → 200 → 100 in two frames is impossible. MAP path stays at 100.
        let mut salience = salience_dominant_at(100, 10);
        let p_other = 0.1_f32 / (PITCH_BINS - 1) as f32;
        let mut outlier = vec![p_other; PITCH_BINS];
        outlier[200] = 0.9;
        salience[5] = outlier;

        let path = most_likely_path(&salience);
        assert_eq!(path, vec![100; 10]);
    }
}
