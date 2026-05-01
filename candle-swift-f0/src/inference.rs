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

pub(crate) fn decode(logits: &Tensor, centers: &Tensor) -> Result<(Vec<f32>, Vec<f32>)> {
    todo!()
}
