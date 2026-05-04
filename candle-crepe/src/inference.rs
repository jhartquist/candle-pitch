use candle_core::{Result, Tensor};

use crate::model::{Crepe, PITCH_BINS};
use crate::viterbi;

// Bin 0 is C1 (~32.7 Hz); bins are 20 cents apart up to bin 359.
#[allow(clippy::excessive_precision)]
const CENTS_OFFSET: f64 = 1997.3794084376191;
const CENTS_PER_BIN: f64 = 7180.0 / (PITCH_BINS as f64 - 1.0);

pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_LENGTH: usize = 1024;
pub const HOP_LENGTH: usize = 160; // 10 ms

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decoder {
    LocalAverage,
    Viterbi,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictionFrame {
    pub time_seconds: f32,
    pub frequency_hz: f32,
    pub confidence: f32,
}

pub(crate) struct Decoded {
    pub frequencies_hz: Vec<f32>,
    pub confidences: Vec<f32>,
}

pub fn predict(model: &Crepe, audio: &[f32], decoder: Decoder) -> Result<Vec<PredictionFrame>> {
    let salience = model.salience(audio)?;
    let decoded = decode(&salience, decoder)?;
    let time_step = HOP_LENGTH as f32 / SAMPLE_RATE as f32;
    Ok(decoded
        .frequencies_hz
        .iter()
        .zip(&decoded.confidences)
        .enumerate()
        .map(|(i, (&frequency_hz, &confidence))| PredictionFrame {
            time_seconds: i as f32 * time_step,
            frequency_hz,
            confidence,
        })
        .collect())
}

pub(crate) fn decode(salience: &Tensor, decoder: Decoder) -> Result<Decoded> {
    let salience_rows: Vec<Vec<f32>> = salience.to_vec2()?;
    let centers = match decoder {
        Decoder::Viterbi => Some(viterbi::most_likely_path(&salience_rows)),
        Decoder::LocalAverage => None,
    };
    let cents = local_average_cents(&salience_rows, centers.as_deref());
    let frequencies_hz = cents
        .iter()
        .map(|&c| {
            if c.is_nan() {
                0.0
            } else {
                (10.0 * 2f64.powf(c / 1200.0)) as f32
            }
        })
        .collect();
    let confidences = salience.max(1)?.to_vec1::<f32>()?;
    Ok(Decoded {
        frequencies_hz,
        confidences,
    })
}

fn local_average_cents(salience: &[Vec<f32>], centers: Option<&[usize]>) -> Vec<f64> {
    let n_frames = salience.len();
    let mut cents = Vec::with_capacity(n_frames);
    for t in 0..n_frames {
        let row = &salience[t];
        let center = match centers {
            Some(c) => c[t],
            None => argmax_first(row),
        };
        let lo = center.saturating_sub(4);
        let hi = (center + 5).min(PITCH_BINS);
        let weights = &row[lo..hi];
        let sum_w: f32 = weights.iter().sum();
        let mut sum_wc = 0.0_f64;
        for (offset, &w) in weights.iter().enumerate() {
            let b = lo + offset;
            sum_wc += w as f64 * (CENTS_OFFSET + b as f64 * CENTS_PER_BIN);
        }
        cents.push(sum_wc / sum_w as f64);
    }
    cents
}

pub(crate) fn argmax_first<T: PartialOrd + Copy>(row: &[T]) -> usize {
    let mut best = 0;
    for (i, v) in row.iter().enumerate().skip(1) {
        if *v > row[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Capacity;
    use crate::test_helpers::{load_fixture, max_cents_diff};
    use candle_core::Device;

    const CAPACITY: Capacity = Capacity::Tiny;
    const TOLERANCE: f32 = 1e-2;

    #[test]
    fn decode_local_average_parity() -> Result<()> {
        let device = Device::Cpu;
        let expected = load_fixture(CAPACITY, &device);
        let decoded = decode(&expected["salience"], Decoder::LocalAverage)?;
        let expected_frequencies: Vec<f32> = expected["frequency_local"].to_vec1()?;
        let expected_confidences: Vec<f32> = expected["confidence"].to_vec1()?;

        let cents_diff = max_cents_diff(&decoded.frequencies_hz, &expected_frequencies);
        assert!(cents_diff < TOLERANCE, "freq cents diff {cents_diff:.2e}");

        let conf_diff = decoded
            .confidences
            .iter()
            .zip(&expected_confidences)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(conf_diff < TOLERANCE, "conf diff {conf_diff:.2e}");
        Ok(())
    }

    #[test]
    fn decode_viterbi_parity() -> Result<()> {
        let device = Device::Cpu;
        let expected = load_fixture(CAPACITY, &device);
        let decoded = decode(&expected["salience"], Decoder::Viterbi)?;
        let expected_frequencies: Vec<f32> = expected["frequency_viterbi"].to_vec1()?;
        let cents_diff = max_cents_diff(&decoded.frequencies_hz, &expected_frequencies);
        assert!(
            cents_diff < TOLERANCE,
            "viterbi cents diff {cents_diff:.2e}"
        );
        Ok(())
    }
}
