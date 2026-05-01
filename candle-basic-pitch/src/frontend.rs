use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::weights::load_safetensors;

pub const N_BINS: usize = 309;

pub struct Frontend {
    kernel_real: Vec<f32>,
    kernel_imag: Vec<f32>,
    lowpass: Vec<f32>,
    lengths_sqrt: Vec<f32>,
    n_octaves: usize,
    n_filters: usize,
    n_fft: usize,
    hop_length: usize,
    downsample_factor: f32,
    device: Device,
}

impl Frontend {
    pub fn from_safetensors(bytes: &[u8], device: &Device) -> Result<Self> {
        Self::new(load_safetensors(bytes, device)?)
    }

    pub fn new(vb: VarBuilder) -> Result<Self> {
        todo!()
    }

    // returns CQT magnitudes as (1, 1, n_frames, N_BINS).
    pub fn forward(&self, audio: &[f32]) -> Result<Tensor> {
        todo!()
    }
}
