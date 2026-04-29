use candle_core::{Device, Result, Tensor};
use candle_nn::{BatchNorm, Conv1d, Conv1dConfig, Linear, VarBuilder, batch_norm, conv1d, linear};

use crate::inference::{FRAME_LENGTH, HOP_LENGTH};
use crate::weights::load_safetensors;

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
    const ALL: [Self; 5] = [
        Self::Tiny,
        Self::Small,
        Self::Medium,
        Self::Large,
        Self::Full,
    ];

    fn multiplier(self) -> usize {
        match self {
            Self::Tiny => 4,
            Self::Small => 8,
            Self::Medium => 16,
            Self::Large => 24,
            Self::Full => 32,
        }
    }

    fn from_multiplier(multiplier: usize) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|c| c.multiplier() == multiplier)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("unknown crepe capacity multiplier: {multiplier}"))
            })
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
        let conv = conv1d(
            in_channels,
            out_channels,
            kernel,
            Conv1dConfig {
                stride,
                ..Default::default()
            },
            vb.pp("conv"),
        )?;
        let bn = batch_norm(out_channels, BN_EPS, vb.pp("bn"))?;
        Ok(Self { conv, bn, padding })
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
        Self::new(load_safetensors(bytes, device)?)
    }

    pub fn new(vb: VarBuilder) -> Result<Self> {
        let conv1_out = vb.get_unchecked("conv1.conv.weight")?.dim(0)?;
        let multiplier = conv1_out / FILTERS[0];
        let capacity = Capacity::from_multiplier(multiplier)?;

        let [b0, b1, b2, b3, b4, b5] = std::array::from_fn(|i| {
            let in_channels = if i == 0 {
                1
            } else {
                FILTERS[i - 1] * multiplier
            };
            ConvBlock::new(
                in_channels,
                FILTERS[i] * multiplier,
                KERNELS[i],
                STRIDES[i],
                PADDINGS[i],
                vb.pp(format!("conv{}", i + 1)),
            )
        });
        let blocks = [b0?, b1?, b2?, b3?, b4?, b5?];

        let classifier = linear(4 * FILTERS[5] * multiplier, PITCH_BINS, vb.pp("classifier"))?;
        Ok(Self {
            blocks,
            classifier,
            capacity,
            device: vb.device().clone(),
        })
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
        let pad = FRAME_LENGTH / 2;
        let n_frames = 1 + (audio.len() + 2 * pad - FRAME_LENGTH) / HOP_LENGTH;
        let mut buffer = vec![0f32; n_frames * FRAME_LENGTH];
        for i in 0..n_frames {
            let frame_start = i * HOP_LENGTH;
            let dst = &mut buffer[i * FRAME_LENGTH..(i + 1) * FRAME_LENGTH];
            let src_lo = frame_start.saturating_sub(pad);
            let src_hi = (frame_start + FRAME_LENGTH)
                .saturating_sub(pad)
                .min(audio.len());
            if src_lo < src_hi {
                let dst_lo = src_lo + pad - frame_start;
                dst[dst_lo..dst_lo + (src_hi - src_lo)].copy_from_slice(&audio[src_lo..src_hi]);
            }
        }
        let frames = Tensor::from_vec(buffer, (n_frames, FRAME_LENGTH), &self.device)?;
        let mean = frames.mean_keepdim(1)?;
        let centered = frames.broadcast_sub(&mean)?;
        let std = centered
            .sqr()?
            .mean_keepdim(1)?
            .sqrt()?
            .clamp(1e-8, f32::INFINITY)?;
        centered.broadcast_div(&std)?.unsqueeze(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{load_fixture, load_weights, max_abs_diff};

    const CAPACITY: Capacity = Capacity::Tiny;
    const TOLERANCE: f32 = 1e-6;

    #[test]
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
