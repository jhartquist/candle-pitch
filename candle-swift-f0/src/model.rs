use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, VarBuilder, conv1d, conv2d};

use crate::frontend::Frontend;
use crate::weights::load_safetensors;

pub const N_PITCH_BINS: usize = 200;
pub const N_FREQ_BINS: usize = 132;
pub const F_MIN: f32 = 46.875;
pub const F_MAX: f32 = 2093.75;

const CHANNELS: [usize; 5] = [8, 16, 32, 64, 1];
const KERNEL: usize = 5;

struct ConvBlock {
    conv: Conv2d,
}

impl ConvBlock {
    fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv = conv2d(
            in_channels,
            out_channels,
            KERNEL,
            Conv2dConfig {
                padding: KERNEL / 2,
                ..Default::default()
            },
            vb,
        )?;
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)?.relu()
    }
}

pub struct SwiftF0 {
    blocks: Vec<ConvBlock>,
    freq_projection: Conv1d,
    frontend: Frontend,
}

impl SwiftF0 {
    pub fn from_safetensors(bytes: &[u8], device: &Device) -> Result<Self> {
        Self::new(load_safetensors(bytes, device)?)
    }

    pub fn new(vb: VarBuilder) -> Result<Self> {
        let blocks: Vec<ConvBlock> = (0..CHANNELS.len())
            .map(|i| {
                let in_ch = if i == 0 { 1 } else { CHANNELS[i - 1] };
                ConvBlock::new(in_ch, CHANNELS[i], vb.pp(format!("conv{}", i + 1)))
            })
            .collect::<Result<_>>()?;

        let freq_projection = conv1d(
            N_FREQ_BINS,
            N_PITCH_BINS,
            1,
            Conv1dConfig::default(),
            vb.pp("freq_projection"),
        )?;

        Ok(Self {
            blocks,
            freq_projection,
            frontend: Frontend::new(vb.device()),
        })
    }

    pub fn forward(&self, audio: &[f32]) -> Result<Tensor> {
        let mut x = self.frontend.forward(audio)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        // (1, 1, 132, T) -> (1, 132, T) -> conv1d -> (1, 200, T) -> permute to logits (1, T, 200).
        let x = x.squeeze(1)?;
        self.freq_projection.forward(&x)?.permute((0, 2, 1))
    }
}

pub(crate) fn pitch_bin_centers() -> [f32; N_PITCH_BINS] {
    let delta = (F_MAX / F_MIN).log2() / (N_PITCH_BINS as f32 - 1.0);
    std::array::from_fn(|i| F_MIN * 2f32.powf(i as f32 * delta))
}
