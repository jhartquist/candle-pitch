use candle_core::{Device, Result, Tensor};
use candle_nn::{Conv1d, Conv2d, VarBuilder};

pub const N_PITCH_BINS: usize = 200;
pub const N_FREQ_BINS: usize = 132;
pub const F_MIN: f32 = 46.875;
pub const F_MAX: f32 = 2093.75;

const CHANNELS: [usize; 5] = [8, 16, 32, 64, 1];
const KERNEL: usize = 5;
const K_MIN: usize = 3;

struct ConvBlock {
    conv: Conv2d,
}

impl ConvBlock {
    fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        todo!()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

pub struct SwiftF0 {
    blocks: [ConvBlock; 5],
    freq_projection: Conv1d,
    window: Tensor,
    pitch_bin_centers: Tensor,
    device: Device,
}

impl SwiftF0 {
    pub fn from_safetensors(bytes: &[u8], device: &Device) -> Result<Self> {
        todo!()
    }

    pub fn new(vb: VarBuilder) -> Result<Self> {
        todo!()
    }

    pub fn forward(&self, audio: &[f32]) -> Result<Tensor> {
        todo!()
    }

    fn frontend(&self, audio: &[f32]) -> Result<Tensor> {
        todo!()
    }
}
