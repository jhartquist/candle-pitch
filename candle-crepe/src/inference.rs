use candle_core::{Result, Tensor};

use crate::model::Crepe;

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

pub fn predict(model: &Crepe, audio: &[f32], decoder: Decoder) -> Result<Vec<PredictionFrame>> {
    todo!()
}

pub(crate) fn decode(salience: &Tensor, decoder: Decoder) -> Result<(Vec<f32>, Vec<f32>)> {
    todo!()
}

fn local_average_cents(salience: &Tensor, centers: Option<&[usize]>) -> Result<Vec<f32>> {
    todo!()
}

fn viterbi_path(salience: &Tensor) -> Result<Vec<usize>> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Capacity;
    use crate::test_helpers::{load_fixture, max_cents_diff};
    use candle_core::Device;

    const CAPACITY: Capacity = Capacity::Tiny;
    const TOLERANCE: f32 = 1e-4;

    #[test]
    #[ignore]
    fn decode_local_average_parity() -> Result<()> {
        let device = Device::Cpu;
        let expected = load_fixture(CAPACITY, &device);
        let (frequencies, confidences) = decode(&expected["salience"], Decoder::LocalAverage)?;
        let expected_frequencies: Vec<f32> = expected["frequency_local"].to_vec1()?;
        let expected_confidences: Vec<f32> = expected["confidence"].to_vec1()?;

        let cents_diff = max_cents_diff(&frequencies, &expected_frequencies);
        assert!(cents_diff < TOLERANCE, "freq cents diff {cents_diff:.2e}");

        let conf_diff = confidences
            .iter()
            .zip(&expected_confidences)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(conf_diff < TOLERANCE, "conf diff {conf_diff:.2e}");
        Ok(())
    }

    #[test]
    #[ignore]
    fn decode_viterbi_parity() -> Result<()> {
        let device = Device::Cpu;
        let expected = load_fixture(CAPACITY, &device);
        let (frequencies, _) = decode(&expected["salience"], Decoder::Viterbi)?;
        let expected_frequencies: Vec<f32> = expected["frequency_viterbi"].to_vec1()?;
        let cents_diff = max_cents_diff(&frequencies, &expected_frequencies);
        assert!(
            cents_diff < TOLERANCE,
            "viterbi cents diff {cents_diff:.2e}"
        );
        Ok(())
    }
}
