use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::weights::load_safetensors;

pub const N_BINS: usize = 309;

pub struct Frontend {
    // top-octave conv kernels, row-major (n_filters, n_fft). reused across
    // every octave with progressively halved hop & downsampled audio.
    kernel_real: Vec<f32>,
    kernel_imag: Vec<f32>,
    // inter-octave anti-aliasing FIR; applied before each downsample-by-2.
    lowpass: Vec<f32>,
    // per-bin sqrt(kernel_length); applied at the end so the magnitudes match
    // librosa's normalization (the only path basic-pitch trains against).
    lengths_sqrt: Vec<f32>,
    n_octaves: usize,
    n_filters: usize,
    n_fft: usize,
    hop_length: usize,
    // early-downsample correction. 1.0 for our params (no early downsample).
    downsample_factor: f32,
    device: Device,
}

impl Frontend {
    pub fn from_safetensors(bytes: &[u8], device: &Device) -> Result<Self> {
        Self::new(load_safetensors(bytes, device)?)
    }

    pub fn new(vb: VarBuilder) -> Result<Self> {
        let kernel_real = tensor_to_vec(&vb, "cqt.kernel_real")?;
        let kernel_imag = tensor_to_vec(&vb, "cqt.kernel_imag")?;
        let lowpass = tensor_to_vec(&vb, "cqt.lowpass")?;
        let lengths = tensor_to_vec(&vb, "cqt.lengths")?;

        let kr = vb.get_unchecked("cqt.kernel_real")?;
        let (n_filters, n_fft) = kr.dims2()?;

        let n_octaves = scalar_usize(&vb, "cqt.n_octaves")?;
        let n_bins = scalar_usize(&vb, "cqt.n_bins")?;
        let hop_length = scalar_usize(&vb, "cqt.hop_length")?;
        let downsample_factor = scalar_f32(&vb, "cqt.downsample_factor")?;
        assert_eq!(scalar_usize(&vb, "cqt.n_fft")?, n_fft);
        assert_eq!(n_bins, N_BINS);
        assert_eq!(lengths.len(), N_BINS);
        assert_eq!(kernel_imag.len(), kernel_real.len());

        let lengths_sqrt = lengths.iter().map(|&l| l.sqrt()).collect();

        Ok(Self {
            kernel_real,
            kernel_imag,
            lowpass,
            lengths_sqrt,
            n_octaves,
            n_filters,
            n_fft,
            hop_length,
            downsample_factor,
            device: vb.device().clone(),
        })
    }

    // returns CQT magnitudes as (1, 1, n_frames, N_BINS).
    pub fn forward(&self, audio: &[f32]) -> Result<Tensor> {
        let mag = self.magnitudes(audio);
        let n_frames = mag.len() / N_BINS;
        Tensor::from_vec(mag, (1, 1, n_frames, N_BINS), &self.device)
    }

    // pure-Rust core. each octave halves both audio length and hop, keeping
    // n_frames constant across octaves (modulo integer-division rounding).
    fn magnitudes(&self, audio: &[f32]) -> Vec<f32> {
        let n_frames = self.frames_at_hop(audio.len(), self.hop_length);
        let total_filters = self.n_octaves * self.n_filters;

        let mut octave_re = vec![0f32; total_filters * n_frames];
        let mut octave_im = vec![0f32; total_filters * n_frames];

        let mut x_down = audio.to_vec();
        let mut hop = self.hop_length;
        for octave in 0..self.n_octaves {
            // nnAudio prepends each new (lower) octave to the front of the
            // accumulator: stacked = concat(new_lower, old). we write directly
            // into the slot for the highest-index (top) octave on iteration 0
            // and shift down on subsequent iterations.
            let dst_octave = self.n_octaves - 1 - octave;
            let dst_lo = dst_octave * self.n_filters * n_frames;
            let dst_hi = dst_lo + self.n_filters * n_frames;
            self.cqt_one_octave(
                &x_down,
                hop,
                n_frames,
                &mut octave_re[dst_lo..dst_hi],
                &mut octave_im[dst_lo..dst_hi],
            );

            if octave + 1 < self.n_octaves {
                x_down = self.lowpass_downsample_by_2(&x_down);
                hop /= 2;
            }
        }

        // trim the leading (lowest, partially-used) bins so we keep exactly
        // N_BINS rows from the top.
        let trim_start = total_filters - N_BINS;
        let scale = self.downsample_factor;

        let mut magnitudes = vec![0f32; n_frames * N_BINS];
        for bin in 0..N_BINS {
            let src_filter = trim_start + bin;
            let row_re = &octave_re[src_filter * n_frames..(src_filter + 1) * n_frames];
            let row_im = &octave_im[src_filter * n_frames..(src_filter + 1) * n_frames];
            let bin_scale = scale * self.lengths_sqrt[bin];
            for frame in 0..n_frames {
                let re = row_re[frame] * bin_scale;
                let im = row_im[frame] * bin_scale;
                magnitudes[frame * N_BINS + bin] = (re * re + im * im).sqrt();
            }
        }
        magnitudes
    }

    // reflect-pad with n_fft/2 each side, then for each frame compute the
    // complex inner product against every kernel row. stride = hop; inputs
    // and kernels both length n_fft.
    fn cqt_one_octave(
        &self,
        audio: &[f32],
        hop: usize,
        n_frames: usize,
        out_re: &mut [f32],
        out_im: &mut [f32],
    ) {
        let pad = self.n_fft / 2;
        let padded = reflect_pad(audio, pad);
        for frame in 0..n_frames {
            let frame_start = frame * hop;
            let window = &padded[frame_start..frame_start + self.n_fft];
            for f in 0..self.n_filters {
                let kr_offset = f * self.n_fft;
                let mut sum_re = 0f32;
                let mut sum_im = 0f32;
                for j in 0..self.n_fft {
                    sum_re += window[j] * self.kernel_real[kr_offset + j];
                    sum_im += window[j] * self.kernel_imag[kr_offset + j];
                }
                let idx = f * n_frames + frame;
                out_re[idx] = sum_re;
                out_im[idx] = sum_im;
            }
        }
    }

    // symmetric zero-pad by (lp_len-1)/2, conv with the lowpass FIR,
    // stride 2 → output length ≈ ceil(input_len / 2).
    fn lowpass_downsample_by_2(&self, audio: &[f32]) -> Vec<f32> {
        let lpf_len = self.lowpass.len();
        let pad = (lpf_len - 1) / 2;
        let padded_len = audio.len() + 2 * pad;
        let mut padded = vec![0f32; padded_len];
        padded[pad..pad + audio.len()].copy_from_slice(audio);

        let n_out = (padded_len - lpf_len) / 2 + 1;
        let mut out = vec![0f32; n_out];
        for (i, slot) in out.iter_mut().enumerate() {
            let start = i * 2;
            let mut sum = 0f32;
            for j in 0..lpf_len {
                sum += padded[start + j] * self.lowpass[j];
            }
            *slot = sum;
        }
        out
    }

    fn frames_at_hop(&self, audio_len: usize, hop: usize) -> usize {
        // reflect-pad of n_fft/2 each side gives audio_len + n_fft total length.
        // with a length-n_fft kernel and stride=hop the conv yields
        // (audio_len + n_fft - n_fft) / hop + 1 = audio_len/hop + 1 frames.
        audio_len / hop + 1
    }
}

fn tensor_to_vec(vb: &VarBuilder, name: &str) -> Result<Vec<f32>> {
    vb.get_unchecked(name)?.flatten_all()?.to_vec1::<f32>()
}

fn scalar_f32(vb: &VarBuilder, name: &str) -> Result<f32> {
    let v = vb.get_unchecked(name)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(v.len(), 1, "{name}: expected length-1, got {}", v.len());
    Ok(v[0])
}

fn scalar_usize(vb: &VarBuilder, name: &str) -> Result<usize> {
    Ok(scalar_f32(vb, name)? as usize)
}

fn reflect_pad(audio: &[f32], pad: usize) -> Vec<f32> {
    // PyTorch / numpy 'reflect': mirror without duplicating the boundary
    // sample. for [a, b, c] with pad=2 -> [c, b, a, b, c, b, a].
    let len = audio.len() as isize;
    let mut out = vec![0f32; audio.len() + 2 * pad];
    out[pad..pad + audio.len()].copy_from_slice(audio);
    for i in 0..pad {
        out[pad - 1 - i] = audio[reflect_index(-(i as isize) - 1, len)];
        out[pad + audio.len() + i] = audio[reflect_index(len + i as isize, len)];
    }
    out
}

fn reflect_index(i: isize, len: isize) -> usize {
    let period = 2 * (len - 1);
    let mut idx = ((i % period) + period) % period;
    if idx >= len {
        idx = period - idx;
    }
    idx as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{load_cqt_weights, load_fixture, max_abs_diff};

    #[test]
    fn cqt_window_parity() -> Result<()> {
        let device = Device::Cpu;
        let frontend = Frontend::from_safetensors(&load_cqt_weights(), &device)?;
        let fixture = load_fixture(&device);
        let audio: Vec<f32> = fixture["window_audio"].to_vec1()?;
        let mag = frontend.forward(&audio)?.squeeze(0)?.squeeze(0)?;
        let diff = max_abs_diff(&mag, &fixture["window_cqt_mag"])?;
        assert!(diff < 1e-3, "cqt diff {diff:.2e}");
        Ok(())
    }
}
