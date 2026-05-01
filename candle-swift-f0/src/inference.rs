use candle_core::{Result, Tensor};

use crate::model::SwiftF0;

pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_LENGTH: usize = 1024;
pub const HOP_LENGTH: usize = 256;
pub const STFT_PADDING: usize = 384;

// Center of first STFT frame in original audio: (FRAME_LENGTH - 1) / 2 - STFT_PADDING.
pub(crate) const CENTER_OFFSET: f32 = (FRAME_LENGTH as f32 - 1.0) / 2.0 - STFT_PADDING as f32;

const PEAK_WIDTH: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictionFrame {
    pub time_seconds: f32,
    pub pitch_hz: f32,
    pub confidence: f32,
}

pub fn predict(model: &SwiftF0, audio: &[f32]) -> Result<Vec<PredictionFrame>> {
    todo!()
}

pub(crate) fn decode(logits: &Tensor) -> Result<(Vec<f32>, Vec<f32>)> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{load_fixture, max_cents_diff};
    use candle_core::Device;

    const TOLERANCE: f32 = 1e-4;

    #[test]
    #[ignore]
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
        assert!(cents_diff < TOLERANCE, "pitch cents diff {cents_diff:.2e}");
        let conf_diff = confidence
            .iter()
            .zip(expected_conf)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(conf_diff < TOLERANCE, "conf diff {conf_diff:.2e}");
        Ok(())
    }
}
