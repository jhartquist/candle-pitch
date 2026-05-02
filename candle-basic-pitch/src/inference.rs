use candle_core::{Result, Tensor};

use crate::frontend::Frontend;
use crate::model::BasicPitch;
use crate::postprocess::Note;

pub const SAMPLE_RATE: u32 = 22_050;
pub const HOP: usize = 256;
pub const ANNOTATIONS_FPS: u32 = SAMPLE_RATE / HOP as u32;

// each window covers ~2 seconds of audio.
pub const AUDIO_N_SAMPLES: usize = 43_844;

pub(crate) const OVERLAP_FRAMES: usize = 30;
pub(crate) const OVERLAP_SAMPLES: usize = OVERLAP_FRAMES * HOP;
pub(crate) const HOP_SAMPLES: usize = AUDIO_N_SAMPLES - OVERLAP_SAMPLES;

// frames trimmed from each end of every window's output before concat, so
// 30-frame overlapping windows align without double-counting.
const N_OLAP_TRIM: usize = OVERLAP_FRAMES / 2;

pub struct Heads {
    pub contour: Tensor,
    pub note: Tensor,
    pub onset: Tensor,
}

pub fn run(model: &BasicPitch, frontend: &Frontend, audio: &[f32]) -> Result<Heads> {
    let original_len = audio.len();

    // prepend OVERLAP_SAMPLES/2 zeros so the first window's center lands at
    // t=0, matching basic_pitch.inference.run_inference.
    let mut padded = vec![0f32; OVERLAP_SAMPLES / 2 + original_len];
    padded[OVERLAP_SAMPLES / 2..].copy_from_slice(audio);

    let mut contour_windows: Vec<Tensor> = Vec::new();
    let mut note_windows: Vec<Tensor> = Vec::new();
    let mut onset_windows: Vec<Tensor> = Vec::new();

    let mut start = 0;
    while start < padded.len() {
        let end = (start + AUDIO_N_SAMPLES).min(padded.len());
        let mut window = padded[start..end].to_vec();
        if window.len() < AUDIO_N_SAMPLES {
            window.resize(AUDIO_N_SAMPLES, 0.0);
        }

        let spec = frontend.forward(&window)?;
        let (contour, note, onset) = model.forward(&spec)?;
        // squeeze (1, 1, T, F) -> (T, F).
        contour_windows.push(contour.squeeze(0)?.squeeze(0)?);
        note_windows.push(note.squeeze(0)?.squeeze(0)?);
        onset_windows.push(onset.squeeze(0)?.squeeze(0)?);

        start += HOP_SAMPLES;
    }

    let n_expected =
        (original_len as f64 * ANNOTATIONS_FPS as f64 / SAMPLE_RATE as f64).floor() as usize;
    Ok(Heads {
        contour: assemble(&contour_windows, n_expected)?,
        note: assemble(&note_windows, n_expected)?,
        onset: assemble(&onset_windows, n_expected)?,
    })
}

pub fn predict(_model: &BasicPitch, _frontend: &Frontend, _audio: &[f32]) -> Result<Vec<Note>> {
    todo!()
}

// trim N_OLAP_TRIM frames from each end of every per-window tensor, concat
// along the time axis, then truncate to n_expected frames so the output
// length matches floor(audio_len * fps / sr) exactly.
fn assemble(windows: &[Tensor], n_expected: usize) -> Result<Tensor> {
    let trimmed: Vec<Tensor> = windows
        .iter()
        .map(|w| {
            let t = w.dim(0)?;
            w.narrow(0, N_OLAP_TRIM, t - 2 * N_OLAP_TRIM)
        })
        .collect::<Result<_>>()?;
    let stacked = Tensor::cat(&trimmed, 0)?;
    let total = stacked.dim(0)?;
    stacked.narrow(0, 0, n_expected.min(total))
}
