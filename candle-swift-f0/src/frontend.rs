use std::sync::Arc;

use candle_core::{Device, Result, Tensor};
use realfft::{RealFftPlanner, RealToComplex};

use crate::inference::{FRAME_LENGTH, HOP_LENGTH, STFT_PADDING};
use crate::model::FREQ_BINS;

// first STFT bin we keep; bins 0..K_MIN are below the model's lowest-frequency band.
const K_MIN: usize = 3;
const LOG_EPS: f32 = 1e-8;

pub(crate) struct Frontend {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    device: Device,
}

impl Frontend {
    pub(crate) fn new(device: &Device) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_LENGTH);
        Self {
            fft,
            window: build_hann_window(),
            device: device.clone(),
        }
    }

    pub(crate) fn forward(&self, audio: &[f32]) -> Result<Tensor> {
        // zero-pad ±STFT_PADDING. matches the bundled ONNX (mode="constant"); see
        // pytorch_parity.py for the upstream-vs-bundled discrepancy.
        let pad = STFT_PADDING;
        let n_frames = (audio.len() + 2 * pad - FRAME_LENGTH) / HOP_LENGTH + 1;

        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut log_mag = vec![0f32; n_frames * FREQ_BINS];

        for t in 0..n_frames {
            // copy frame into the FFT input scratch buffer, treating out-of-range
            // samples (i.e. the front/back zero-padded region) as zero.
            input.fill(0.0);
            let frame_start = t * HOP_LENGTH;
            let src_lo = frame_start.saturating_sub(pad);
            let src_hi = (frame_start + FRAME_LENGTH)
                .saturating_sub(pad)
                .min(audio.len());
            if src_lo < src_hi {
                let dst_lo = src_lo + pad - frame_start;
                input[dst_lo..dst_lo + (src_hi - src_lo)].copy_from_slice(&audio[src_lo..src_hi]);
            }
            for (x, w) in input.iter_mut().zip(&self.window) {
                *x *= w;
            }
            self.fft
                .process(&mut input, &mut spectrum)
                .expect("realfft: input/output buffer mismatch");
            let row = &mut log_mag[t * FREQ_BINS..(t + 1) * FREQ_BINS];
            for (bin, slot) in row.iter_mut().enumerate() {
                let c = spectrum[K_MIN + bin];
                let mag = (c.re * c.re + c.im * c.im).sqrt();
                *slot = (mag + LOG_EPS).ln();
            }
        }

        // (n_frames, FREQ_BINS) -> (1, 1, FREQ_BINS, n_frames) to feed Conv2d.
        Tensor::from_vec(log_mag, (n_frames, FREQ_BINS), &self.device)?
            .t()?
            .unsqueeze(0)?
            .unsqueeze(0)
    }
}

fn build_hann_window() -> Vec<f32> {
    // periodic hann (matches torch.hann_window default and the ONNX initializer).
    let n = FRAME_LENGTH as f32;
    (0..FRAME_LENGTH)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n).cos())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{load_fixture, max_abs_diff};
    use candle_core::D;

    const SLICE_FRAMES: usize = 16;

    #[test]
    fn frontend_parity() -> Result<()> {
        let device = Device::Cpu;
        let expected = load_fixture(&device);
        let frontend = Frontend::new(&device);
        let audio: Vec<f32> = expected["audio_16k"].to_vec1()?;
        let slice_start = expected["slice_start"].to_scalar::<i64>()? as usize;
        let log_mag = frontend
            .forward(&audio)?
            .narrow(D::Minus1, slice_start, SLICE_FRAMES)?;
        // pre-log magnitudes agree with torch.stft to ~1e-5 (fp32 fft noise floor);
        // log(mag + eps) amplifies near-zero bins, so ~1e-3 is the irreducible bound
        // here regardless of the fft implementation.
        let diff = max_abs_diff(&log_mag, &expected["log_mag"])?;
        assert!(diff < 2e-3, "log_mag diff {diff:.2e}");
        Ok(())
    }
}
