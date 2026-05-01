use candle_core::{Result, Tensor};

use crate::frontend::Frontend;
use crate::model::BasicPitch;
use crate::postprocess::Note;

pub const SAMPLE_RATE: u32 = 22_050;
pub const HOP: usize = 256;
pub const ANNOTATIONS_FPS: u32 = SAMPLE_RATE / HOP as u32;

// each window covers ~2 seconds of audio.
pub const AUDIO_N_SAMPLES: usize = 43_844;

pub(crate) const OVERLAP_FRAMES: usize = 30;
pub(crate) const OVERLAP_SAMPLES: usize = OVERLAP_FRAMES * HOP;
pub(crate) const HOP_SAMPLES: usize = AUDIO_N_SAMPLES - OVERLAP_SAMPLES;

pub struct Heads {
    pub contour: Tensor,
    pub note: Tensor,
    pub onset: Tensor,
}

pub fn run(model: &BasicPitch, frontend: &Frontend, audio: &[f32]) -> Result<Heads> {
    todo!()
}

pub fn predict(model: &BasicPitch, frontend: &Frontend, audio: &[f32]) -> Result<Vec<Note>> {
    todo!()
}
