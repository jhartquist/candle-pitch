use candle_core::{Device, Result, Tensor};
use candle_nn::{BatchNorm, Conv2d, VarBuilder};

use crate::weights::load_safetensors;

pub const N_OUTPUT_FREQS: usize = 264;
pub const N_PIANO_KEYS: usize = 88;
pub const N_HARMONICS: usize = 8;

// shift = round(BPO * log2(h)) for harmonics [0.5, 1, 2, 3, 4, 5, 6, 7] with BPO = 36.
const HARMONIC_SHIFTS: [i32; N_HARMONICS] = [-36, 0, 36, 57, 72, 84, 93, 101];

pub struct BasicPitch {
    bn1: BatchNorm,
    contour_conv1: Conv2d,
    bn2: BatchNorm,
    contour_conv2: Conv2d,
    note_conv1: Conv2d,
    note_conv2: Conv2d,
    onset_conv1: Conv2d,
    bn3: BatchNorm,
    onset_conv2: Conv2d,
    device: Device,
}

impl BasicPitch {
    pub fn from_safetensors(bytes: &[u8], device: &Device) -> Result<Self> {
        Self::new(load_safetensors(bytes, device)?)
    }

    pub fn new(vb: VarBuilder) -> Result<Self> {
        todo!()
    }

    // takes magnitudes of shape (1, 1, T, N_BINS) and returns
    // (contour, note, onset).
    pub fn forward(&self, spec: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        todo!()
    }
}
