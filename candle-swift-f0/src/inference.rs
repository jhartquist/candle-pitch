use candle_core::{Result, Tensor};
use candle_nn::ops;

use crate::model::{SwiftF0, pitch_bin_centers};

pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_LENGTH: usize = 1024;
pub const HOP_LENGTH: usize = 256;
pub const STFT_PADDING: usize = 384;

// center of first STFT frame in original audio: (FRAME_LENGTH - 1) / 2 - STFT_PADDING.
pub(crate) const CENTER_OFFSET: f32 = (FRAME_LENGTH as f32 - 1.0) / 2.0 - STFT_PADDING as f32;

const PEAK_WIDTH: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictionFrame {
    pub time_seconds: f32,
    pub pitch_hz: f32,
    pub confidence: f32,
}

pub fn predict(model: &SwiftF0, audio: &[f32]) -> Result<Vec<PredictionFrame>> {
    let logits = model.forward(audio)?;
    let (pitch_hz, confidence) = decode(&logits)?;
    Ok(pitch_hz
        .iter()
        .zip(&confidence)
        .enumerate()
        .map(|(i, (&pitch_hz, &confidence))| PredictionFrame {
            time_seconds: (i as f32 * HOP_LENGTH as f32 + CENTER_OFFSET) / SAMPLE_RATE as f32,
            pitch_hz,
            confidence,
        })
        .collect())
}

pub(crate) fn decode(logits: &Tensor) -> Result<(Vec<f32>, Vec<f32>)> {
    // logits (1, T, N=200) -> softmax over the bin axis, then per-frame the ±PEAK_WIDTH
    // window around the argmax: confidence = sum(masked probs); pitch_hz = expectation
    // of the corresponding bin centers under those masked probs.
    let probs: Vec<Vec<f32>> = ops::softmax_last_dim(&logits.contiguous()?)?
        .squeeze(0)?
        .to_vec2()?;
    let centers = pitch_bin_centers();
    let mut pitch_hz = Vec::with_capacity(probs.len());
    let mut confidence = Vec::with_capacity(probs.len());
    for frame in &probs {
        let peak = argmax_first(frame);
        let lo = peak.saturating_sub(PEAK_WIDTH);
        let hi = (peak + PEAK_WIDTH + 1).min(frame.len());
        let mut conf = 0.0_f32;
        let mut hz_sum = 0.0_f32;
        for (offset, &p) in frame[lo..hi].iter().enumerate() {
            conf += p;
            hz_sum += p * centers[lo + offset];
        }
        confidence.push(conf);
        pitch_hz.push(hz_sum / (conf + 1e-8));
    }
    Ok((pitch_hz, confidence))
}

fn argmax_first<T: PartialOrd + Copy>(row: &[T]) -> usize {
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
    use crate::test_helpers::{load_fixture, max_cents_diff};
    use candle_core::Device;

    const TOLERANCE: f32 = 1e-4;

    #[test]
    fn decode_parity() -> Result<()> {
        let device = Device::Cpu;
        let expected = load_fixture(&device);
        let (pitch_hz, confidence) = decode(&expected["freq_projection"])?;

        let slice_start = expected["slice_start"].to_scalar::<i64>()? as usize;
        let expected_pitch: Vec<f32> = expected["pitch_hz"].to_vec1()?;
        let expected_conf: Vec<f32> = expected["confidence"].to_vec1()?;
        let expected_pitch = &expected_pitch[slice_start..slice_start + 16];
        let expected_conf = &expected_conf[slice_start..slice_start + 16];

        let cents_diff = max_cents_diff(&pitch_hz, expected_pitch);
        assert!(cents_diff < 1e-3, "pitch cents diff {cents_diff:.2e}");
        let conf_diff = confidence
            .iter()
            .zip(expected_conf)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(conf_diff < TOLERANCE, "conf diff {conf_diff:.2e}");
        Ok(())
    }
}
