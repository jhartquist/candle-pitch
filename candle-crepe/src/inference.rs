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
