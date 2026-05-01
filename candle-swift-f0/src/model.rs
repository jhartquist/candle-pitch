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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{load_fixture, load_weights, max_abs_diff};
    use candle_core::D;

    const TOLERANCE: f32 = 1e-4;
    const SLICE_FRAMES: usize = 16;

    #[test]
    #[ignore]
    fn frontend_parity() -> Result<()> {
        let device = Device::Cpu;
        let expected = load_fixture(&device);
        let model = SwiftF0::from_safetensors(&load_weights(), &device)?;
        let audio: Vec<f32> = expected["audio_16k"].to_vec1()?;
        let slice_start = expected["slice_start"].to_scalar::<i64>()? as usize;
        let log_mag = model
            .frontend(&audio)?
            .narrow(D::Minus1, slice_start, SLICE_FRAMES)?;
        let diff = max_abs_diff(&log_mag, &expected["log_mag"])?;
        assert!(diff < TOLERANCE, "log_mag diff {diff:.2e}");
        Ok(())
    }
}
