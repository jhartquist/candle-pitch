use candle_core::{Device, Result, Tensor};
use candle_nn::{BatchNorm, Conv1d, Linear, VarBuilder};

pub const PITCH_BINS: usize = 360;

const FILTERS: [usize; 6] = [32, 4, 4, 4, 8, 16];
const KERNELS: [usize; 6] = [512, 64, 64, 64, 64, 64];
const STRIDES: [usize; 6] = [4, 1, 1, 1, 1, 1];
const PADDINGS: [(usize, usize); 6] =
    [(254, 254), (31, 32), (31, 32), (31, 32), (31, 32), (31, 32)];
const BN_EPS: f64 = 1e-3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capacity {
    Tiny,
    Small,
    Medium,
    Large,
    Full,
}

impl Capacity {
    fn multiplier(self) -> usize {
        todo!()
    }
}

impl std::fmt::Display for Capacity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
            Self::Full => "full",
        })
    }
}

struct ConvBlock {
    conv: Conv1d,
    bn: BatchNorm,
    padding: (usize, usize),
}

impl ConvBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: (usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        todo!()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

pub struct Crepe {
    blocks: [ConvBlock; 6],
    classifier: Linear,
    capacity: Capacity,
    device: Device,
}

impl Crepe {
    pub fn from_safetensors(bytes: &[u8], device: &Device) -> Result<Self> {
        todo!()
    }

    pub fn new(vb: VarBuilder) -> Result<Self> {
        todo!()
    }

    pub fn capacity(&self) -> Capacity {
        self.capacity
    }

    pub fn forward(&self, frames: &Tensor) -> Result<Tensor> {
        todo!()
    }

    pub fn salience(&self, audio: &[f32]) -> Result<Tensor> {
        todo!()
    }

    fn frame(&self, audio: &[f32]) -> Result<Tensor> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{load_fixture, load_weights, max_abs_diff};

    const CAPACITY: Capacity = Capacity::Tiny;
    const TOLERANCE: f32 = 1e-4;

    #[test]
    #[ignore]
    fn frame_parity() -> Result<()> {
        let device = Device::Cpu;
        let expected = load_fixture(CAPACITY, &device);
        let model = Crepe::from_safetensors(&load_weights(CAPACITY), &device)?;
        let audio: Vec<f32> = expected["audio_16k"].to_vec1()?;
        let slice_start = expected["slice_start"].to_scalar::<i64>()? as usize;
        let frames = model.frame(&audio)?.narrow(0, slice_start, 16)?;
        let diff = max_abs_diff(&frames, &expected["frames"])?;
        assert!(diff < TOLERANCE, "frames diff {diff:.2e}");
        Ok(())
    }

}
