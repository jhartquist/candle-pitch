use candle_core::{D, Device, ModuleT, Result, Tensor};
use candle_nn::{BatchNorm, VarBuilder, batch_norm, ops};

use crate::frontend::N_BINS;
use crate::weights::load_safetensors;

pub const N_OUTPUT_FREQS: usize = 264;
pub const N_PIANO_KEYS: usize = 88;
pub const N_HARMONICS: usize = 8;

// Keras/TF default; NOT PyTorch's 1e-5.
const BN_EPS: f64 = 1e-3;

// shift = round(BPO * log2(h)) for harmonics [0.5, 1, 2, 3, 4, 5, 6, 7] with BPO = 36.
const HARMONIC_SHIFTS: [i32; N_HARMONICS] = [-36, 0, 36, 57, 72, 84, 93, 101];

pub struct BasicPitch {
    bn1: BatchNorm,
    contour_conv1: ConvWeights,
    bn2: BatchNorm,
    contour_conv2: ConvWeights,
    note_conv1: ConvWeights,
    note_conv2: ConvWeights,
    onset_conv1: ConvWeights,
    bn3: BatchNorm,
    onset_conv2: ConvWeights,
    device: Device,
}

struct ConvWeights {
    weight: Tensor,
    bias: Tensor,
}

impl ConvWeights {
    fn load(vb: &VarBuilder, name: &str) -> Result<Self> {
        Ok(Self {
            weight: vb.get_unchecked(&format!("{name}.weight"))?,
            bias: vb.get_unchecked(&format!("{name}.bias"))?,
        })
    }
}

impl BasicPitch {
    pub fn from_safetensors(bytes: &[u8], device: &Device) -> Result<Self> {
        Self::new(load_safetensors(bytes, device)?)
    }

    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            bn1: batch_norm(1, BN_EPS, vb.pp("bn1"))?,
            contour_conv1: ConvWeights::load(&vb, "contour_conv1")?,
            bn2: batch_norm(8, BN_EPS, vb.pp("bn2"))?,
            contour_conv2: ConvWeights::load(&vb, "contour_conv2")?,
            note_conv1: ConvWeights::load(&vb, "note_conv1")?,
            note_conv2: ConvWeights::load(&vb, "note_conv2")?,
            onset_conv1: ConvWeights::load(&vb, "onset_conv1")?,
            bn3: batch_norm(32, BN_EPS, vb.pp("bn3"))?,
            onset_conv2: ConvWeights::load(&vb, "onset_conv2")?,
            device: vb.device().clone(),
        })
    }

    // takes magnitudes of shape (1, 1, T, N_BINS) and returns
    // (contour, note, onset) each (1, 1, T, *).
    pub fn forward(&self, spec: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let x = normalized_log(spec)?;
        let x = self.bn1.forward_t(&x, false)?;
        let hcqt = harmonic_stacking(&x)?;

        let x_c = conv2d_same(&hcqt, &self.contour_conv1, (1, 1))?;
        let x_c = self.bn2.forward_t(&x_c, false)?;
        let x_c = x_c.relu()?;
        let x_c = conv2d_same(&x_c, &self.contour_conv2, (1, 1))?;
        let contour = ops::sigmoid(&x_c)?;

        let x_n = conv2d_same(&contour, &self.note_conv1, (1, 3))?;
        let x_n = x_n.relu()?;
        let x_n = conv2d_same(&x_n, &self.note_conv2, (1, 1))?;
        let note = ops::sigmoid(&x_n)?;

        let x_o = conv2d_same(&hcqt, &self.onset_conv1, (1, 3))?;
        let x_o = self.bn3.forward_t(&x_o, false)?;
        let x_o = x_o.relu()?;
        let x_o = Tensor::cat(&[&note, &x_o], 1)?;
        let x_o = conv2d_same(&x_o, &self.onset_conv2, (1, 1))?;
        let onset = ops::sigmoid(&x_o)?;

        Ok((contour, note, onset))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

// min-max normalized log-power on the magnitudes' squared values.
// output lives in [0, 1] per call (so callers normalize per window, not per clip).
fn normalized_log(x: &Tensor) -> Result<Tensor> {
    let power = x.sqr()?;
    let log10 = (10.0_f64).ln();
    let log_power = ((power + 1e-10)?.log()? * (10.0 / log10))?;
    let min_val = log_power.flatten_all()?.min(0)?;
    let offset = log_power.broadcast_sub(&min_val)?;
    let max_val = offset.flatten_all()?.max(0)?;
    let max_scalar = max_val.to_scalar::<f32>()?;
    if max_scalar > 0.0 {
        offset.broadcast_div(&max_val)
    } else {
        Ok(offset)
    }
}

// HCQT: stack 8 frequency-shifted copies of the (B, 1, T, N_BINS) input
// along the channel axis, cropping each to N_OUTPUT_FREQS columns.
fn harmonic_stacking(x: &Tensor) -> Result<Tensor> {
    let (_, _, _, n_freqs) = x.dims4()?;
    assert_eq!(n_freqs, N_BINS);
    let mut channels = Vec::with_capacity(N_HARMONICS);
    for &shift in &HARMONIC_SHIFTS {
        let shifted = if shift == 0 {
            x.clone()
        } else if shift > 0 {
            let s = shift as usize;
            x.narrow(D::Minus1, s, n_freqs - s)?
                .pad_with_zeros(D::Minus1, 0, s)?
        } else {
            let s = (-shift) as usize;
            x.narrow(D::Minus1, 0, n_freqs - s)?
                .pad_with_zeros(D::Minus1, s, 0)?
        };
        channels.push(shifted.narrow(D::Minus1, 0, N_OUTPUT_FREQS)?);
    }
    Tensor::cat(&channels, 1)
}

// Keras-style SAME padding with asymmetric stride support. candle's conv2d
// takes a single stride scalar, so for stride (1, 3) we run a stride-1 conv
// then index_select every 3rd column on the width axis.
fn conv2d_same(input: &Tensor, conv: &ConvWeights, stride: (usize, usize)) -> Result<Tensor> {
    let (_, _, h_in, w_in) = input.dims4()?;
    let (c_out, _, k_h, k_w) = conv.weight.dims4()?;
    let (s_h, s_w) = stride;

    let out_h = h_in.div_ceil(s_h);
    let out_w = w_in.div_ceil(s_w);
    let pad_h = ((out_h - 1) * s_h + k_h).saturating_sub(h_in);
    let pad_w = ((out_w - 1) * s_w + k_w).saturating_sub(w_in);

    let mut x = input.clone();
    if pad_h > 0 {
        x = x.pad_with_zeros(2, pad_h / 2, pad_h - pad_h / 2)?;
    }
    if pad_w > 0 {
        x = x.pad_with_zeros(3, pad_w / 2, pad_w - pad_w / 2)?;
    }

    let x = x.conv2d(&conv.weight, 0, 1, 1, 1)?;
    let bias = conv.bias.reshape((1, c_out, 1, 1))?;
    let mut x = x.broadcast_add(&bias)?;

    if s_h > 1 {
        let idx: Vec<u32> = (0..out_h).map(|i| (i * s_h) as u32).collect();
        x = x.index_select(&Tensor::from_vec(idx, out_h, input.device())?, 2)?;
    }
    if s_w > 1 {
        let idx: Vec<u32> = (0..out_w).map(|i| (i * s_w) as u32).collect();
        x = x.index_select(&Tensor::from_vec(idx, out_w, input.device())?, 3)?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{load_fixture, load_weights, max_abs_diff};

    const TOLERANCE: f32 = 5e-5;

    #[test]
    fn forward_window_parity() -> Result<()> {
        let device = Device::Cpu;
        let model = BasicPitch::from_safetensors(&load_weights(), &device)?;
        let fixture = load_fixture(&device);

        // window_cqt_mag is (T, N_BINS); model.forward expects (1, 1, T, N_BINS).
        let spec = fixture["window_cqt_mag"].unsqueeze(0)?.unsqueeze(0)?;
        let (contour, note, onset) = model.forward(&spec)?;

        // model emits (1, 1, T, F); fixture has the batch axis stripped → (1, T, F).
        let contour_diff = max_abs_diff(&contour.squeeze(0)?, &fixture["window_contour"])?;
        let note_diff = max_abs_diff(&note.squeeze(0)?, &fixture["window_note"])?;
        let onset_diff = max_abs_diff(&onset.squeeze(0)?, &fixture["window_onset"])?;
        assert!(contour_diff < TOLERANCE, "contour diff {contour_diff:.2e}");
        assert!(note_diff < TOLERANCE, "note diff {note_diff:.2e}");
        assert!(onset_diff < TOLERANCE, "onset diff {onset_diff:.2e}");
        Ok(())
    }
}
