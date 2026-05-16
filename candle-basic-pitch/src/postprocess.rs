use candle_core::{Result, Tensor};

use crate::inference::{HOP, SAMPLE_RATE};
use crate::model::N_PIANO_KEYS;

// midi 21 (A0) lives in note posteriorgram bin 0.
const MIDI_OFFSET: u8 = 21;

// piano frequency reference for contour-bin <-> midi mapping.
const ANNOTATIONS_BASE_FREQUENCY: f64 = 27.5;
const CONTOURS_BINS_PER_SEMITONE: usize = 3;
const N_FREQ_BINS_CONTOURS: usize = N_PIANO_KEYS * CONTOURS_BINS_PER_SEMITONE;

const ENERGY_TOL: usize = 11;
const N_BINS_TOLERANCE: usize = 25;
const GAUSSIAN_STD: f64 = 5.0;

#[derive(Clone, Debug)]
pub struct Note {
    pub points: Vec<NotePoint>,
}

#[derive(Clone, Copy, Debug)]
pub struct NotePoint {
    pub time: f32,
    pub frequency: f32,
    pub confidence: f32,
}

impl Note {
    pub fn start(&self) -> f32 {
        self.points.first().expect("note has no points").time
    }

    pub fn end(&self) -> f32 {
        self.points.last().expect("note has no points").time
    }
}

// internal event used by the polyphonic detector. frame indices are inclusive
// of `start_frame` and exclusive of `end_frame`. `pitch_bends[i]` is the
// integer offset (in contour bins) of the per-frame argmax around the
// nominal contour bin for `midi`.
#[derive(Clone, Debug)]
struct NoteEvent {
    start_frame: usize,
    end_frame: usize,
    midi: u8,
    amplitude: f32,
    pitch_bends: Vec<i32>,
}

pub(crate) fn detect_notes(
    contour: &Tensor,
    note: &Tensor,
    onset: &Tensor,
    onset_thresh: f32,
    frame_thresh: f32,
    min_note_len: usize,
) -> Result<Vec<Note>> {
    let note_grid: Vec<Vec<f32>> = note.to_vec2()?;
    let onset_grid: Vec<Vec<f32>> = onset.to_vec2()?;
    let contour_grid: Vec<Vec<f32>> = contour.to_vec2()?;

    let mut events = output_to_notes_polyphonic(
        &note_grid,
        &onset_grid,
        onset_thresh,
        frame_thresh,
        min_note_len,
    );
    attach_pitch_bends(&mut events, &contour_grid);
    Ok(events.into_iter().map(event_to_note).collect())
}

// port of basic_pitch.note_creation.output_to_notes_polyphonic.
fn output_to_notes_polyphonic(
    note_grid: &[Vec<f32>],
    onset_grid: &[Vec<f32>],
    onset_thresh: f32,
    frame_thresh: f32,
    min_note_len: usize,
) -> Vec<NoteEvent> {
    let n_frames = note_grid.len();
    let n_freqs = note_grid[0].len();
    let combined_onsets = inferred_onsets(onset_grid, note_grid);
    let mut remaining: Vec<Vec<f32>> = note_grid.to_vec();

    // local maxima on the time axis, latest peaks first (numpy `[::-1]`).
    let mut peaks: Vec<(usize, usize)> = argrelmax_axis0(&combined_onsets)
        .into_iter()
        .filter(|&(t, f)| combined_onsets[t][f] >= onset_thresh)
        .collect();
    peaks.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));

    let mut events = Vec::new();
    for (start, freq_idx) in peaks {
        if start >= n_frames - 1 {
            continue;
        }
        let end = walk_forward(&remaining, start, freq_idx, frame_thresh);
        if end - start <= min_note_len {
            continue;
        }
        // mark this note's energy band consumed so the melodia sweep won't
        // re-pick it (covers start..end inclusive, ±1 in frequency).
        for row in &mut remaining[start..end] {
            zero_band(row, freq_idx, n_freqs);
        }
        events.push(make_event(note_grid, start, end, freq_idx));
    }

    melodia_sweep(
        &mut events,
        &mut remaining,
        note_grid,
        frame_thresh,
        min_note_len,
        n_freqs,
    );
    events
}

// after onset-driven detection, sweep the remaining energy by global-max.
// each pick extends both directions until energy_tol consecutive below-thresh
// frames close it out.
fn melodia_sweep(
    events: &mut Vec<NoteEvent>,
    remaining: &mut [Vec<f32>],
    note_grid: &[Vec<f32>],
    frame_thresh: f32,
    min_note_len: usize,
    n_freqs: usize,
) {
    while let Some((i_mid, freq_idx, value)) = argmax_2d(remaining) {
        if value <= frame_thresh {
            break;
        }
        remaining[i_mid][freq_idx] = 0.0;
        let end = walk_forward_consume(remaining, i_mid, freq_idx, frame_thresh, n_freqs);
        let start = walk_backward_consume(remaining, i_mid, freq_idx, frame_thresh, n_freqs);
        if end <= start || end - start <= min_note_len {
            continue;
        }
        events.push(make_event(note_grid, start, end, freq_idx));
    }
}

// observe-only forward walk. matches upstream's `i < n_frames - 1` bound.
fn walk_forward(remaining: &[Vec<f32>], i_mid: usize, freq_idx: usize, frame_thresh: f32) -> usize {
    let n_frames = remaining.len();
    let mut i = i_mid + 1;
    let mut k = 0usize;
    while i < n_frames - 1 && k < ENERGY_TOL {
        k = if remaining[i][freq_idx] < frame_thresh { k + 1 } else { 0 };
        i += 1;
    }
    i - k
}

// consuming forward walk for the melodia sweep.
fn walk_forward_consume(
    remaining: &mut [Vec<f32>],
    i_mid: usize,
    freq_idx: usize,
    frame_thresh: f32,
    n_freqs: usize,
) -> usize {
    let n_frames = remaining.len();
    let mut i = i_mid + 1;
    let mut k = 0usize;
    while i < n_frames - 1 && k < ENERGY_TOL {
        k = if remaining[i][freq_idx] < frame_thresh { k + 1 } else { 0 };
        zero_band(&mut remaining[i], freq_idx, n_freqs);
        i += 1;
    }
    (i - 1).saturating_sub(k)
}

// consuming backward walk. uses isize so the post-decrement matches upstream
// (which leaves `i = -1` at exit).
fn walk_backward_consume(
    remaining: &mut [Vec<f32>],
    i_mid: usize,
    freq_idx: usize,
    frame_thresh: f32,
    n_freqs: usize,
) -> usize {
    let mut i: isize = i_mid as isize - 1;
    let mut k = 0usize;
    while i > -1 && k < ENERGY_TOL {
        let iu = i as usize;
        k = if remaining[iu][freq_idx] < frame_thresh { k + 1 } else { 0 };
        zero_band(&mut remaining[iu], freq_idx, n_freqs);
        i -= 1;
    }
    (i + 1 + k as isize) as usize
}

fn make_event(note_grid: &[Vec<f32>], start: usize, end: usize, freq_idx: usize) -> NoteEvent {
    let amplitude: f32 =
        (start..end).map(|t| note_grid[t][freq_idx]).sum::<f32>() / (end - start) as f32;
    NoteEvent {
        start_frame: start,
        end_frame: end,
        midi: freq_idx as u8 + MIDI_OFFSET,
        amplitude,
        pitch_bends: Vec::new(),
    }
}

fn zero_band(row: &mut [f32], freq_idx: usize, n_freqs: usize) {
    row[freq_idx] = 0.0;
    if freq_idx + 1 < n_freqs {
        row[freq_idx + 1] = 0.0;
    }
    if freq_idx > 0 {
        row[freq_idx - 1] = 0.0;
    }
}

fn argmax_2d(grid: &[Vec<f32>]) -> Option<(usize, usize, f32)> {
    let mut best: Option<(usize, usize, f32)> = None;
    for (t, row) in grid.iter().enumerate() {
        for (f, &v) in row.iter().enumerate() {
            if best.is_none_or(|(_, _, prev)| v > prev) {
                best = Some((t, f, v));
            }
        }
    }
    best
}

// rescue weak/missed onsets by combining the onset head with positive jumps
// in the note posteriorgram. mirrors basic_pitch.note_creation.get_infered_onsets
// with n_diff = 2.
fn inferred_onsets(onset_grid: &[Vec<f32>], note_grid: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n_frames = note_grid.len();
    let n_freqs = note_grid[0].len();

    let mut frame_diff = vec![vec![0f32; n_freqs]; n_frames];
    for t in 2..n_frames {
        for f in 0..n_freqs {
            let d1 = note_grid[t][f] - note_grid[t - 1][f];
            let d2 = note_grid[t][f] - note_grid[t - 2][f];
            frame_diff[t][f] = d1.min(d2).max(0.0);
        }
    }

    let max_onset = flat_max(onset_grid);
    let max_diff = flat_max(&frame_diff);
    let scale = if max_diff > 0.0 { max_onset / max_diff } else { 0.0 };

    onset_grid
        .iter()
        .zip(&frame_diff)
        .map(|(orow, drow)| {
            orow.iter()
                .zip(drow)
                .map(|(&o, &d)| o.max(d * scale))
                .collect()
        })
        .collect()
}

fn flat_max(grid: &[Vec<f32>]) -> f32 {
    grid.iter()
        .flat_map(|row| row.iter().copied())
        .fold(f32::NEG_INFINITY, f32::max)
}

// scipy.argrelmax(arr, axis=0): peaks where arr[t] > arr[t-1] AND arr[t] > arr[t+1].
fn argrelmax_axis0(data: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let n_frames = data.len();
    if n_frames < 3 {
        return Vec::new();
    }
    let n_freqs = data[0].len();
    let mut peaks = Vec::new();
    for t in 1..n_frames - 1 {
        for f in 0..n_freqs {
            let v = data[t][f];
            if v > data[t - 1][f] && v > data[t + 1][f] {
                peaks.push((t, f));
            }
        }
    }
    peaks
}

// gaussian-weighted argmax in a ±25-bin contour window around each note's
// central contour bin.
fn attach_pitch_bends(events: &mut [NoteEvent], contour_grid: &[Vec<f32>]) {
    let window_length = N_BINS_TOLERANCE * 2 + 1;
    let gaussian = gaussian_window(window_length, GAUSSIAN_STD);

    for event in events.iter_mut() {
        let freq_idx = midi_to_contour_bin(event.midi).round() as i32;
        let freq_start = (freq_idx - N_BINS_TOLERANCE as i32).max(0) as usize;
        let freq_end =
            N_FREQ_BINS_CONTOURS.min((freq_idx + N_BINS_TOLERANCE as i32 + 1) as usize);

        // crop the gaussian to the same window we're scanning (it gets clipped
        // when the central bin is near 0 or N_FREQ_BINS_CONTOURS).
        let gauss_lo = (N_BINS_TOLERANCE as i32 - freq_idx).max(0) as usize;
        let gauss_hi = window_length
            - (freq_idx - (N_FREQ_BINS_CONTOURS as i32 - N_BINS_TOLERANCE as i32 - 1)).max(0)
                as usize;
        let gauss = &gaussian[gauss_lo..gauss_hi];
        let pb_shift = N_BINS_TOLERANCE - gauss_lo;

        event.pitch_bends = (event.start_frame..event.end_frame)
            .map(|t| {
                let row = &contour_grid[t][freq_start..freq_end];
                gaussian_argmax(row, gauss) as i32 - pb_shift as i32
            })
            .collect();
    }
}

fn gaussian_argmax(row: &[f32], gauss: &[f64]) -> usize {
    row.iter()
        .zip(gauss)
        .enumerate()
        .max_by(|(_, (a_val, a_g)), (_, (b_val, b_g))| {
            let a = **a_val as f64 * **a_g;
            let b = **b_val as f64 * **b_g;
            a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn gaussian_window(n: usize, std: f64) -> Vec<f64> {
    let center = (n - 1) as f64 / 2.0;
    (0..n)
        .map(|i| {
            let x = (i as f64 - center) / std;
            (-0.5 * x * x).exp()
        })
        .collect()
}

fn midi_to_hz(midi: u8) -> f32 {
    440.0 * 2f32.powf((midi as f32 - 69.0) / 12.0)
}

fn midi_to_contour_bin(midi: u8) -> f64 {
    12.0 * CONTOURS_BINS_PER_SEMITONE as f64
        * (midi_to_hz(midi) as f64 / ANNOTATIONS_BASE_FREQUENCY).log2()
}

fn event_to_note(event: NoteEvent) -> Note {
    let central_hz = midi_to_hz(event.midi);
    let cents_per_bin = 100.0 / CONTOURS_BINS_PER_SEMITONE as f32;
    let frame_to_seconds = HOP as f32 / SAMPLE_RATE as f32;
    let points = (event.start_frame..event.end_frame)
        .zip(event.pitch_bends)
        .map(|(frame, bend)| {
            let cents = bend as f32 * cents_per_bin;
            NotePoint {
                time: frame as f32 * frame_to_seconds,
                frequency: central_hz * 2f32.powf(cents / 1200.0),
                confidence: event.amplitude,
            }
        })
        .collect();
    Note { points }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::load_fixture;
    use candle_core::Device;

    #[test]
    fn argrelmax_finds_strict_local_maxima() {
        let grid = vec![
            vec![0.1, 0.2],
            vec![0.5, 0.3],
            vec![0.4, 0.8],
            vec![0.6, 0.1],
            vec![0.0, 0.0],
        ];
        // t=1: row[0]=0.5 is peak (0.1<0.5>0.4); row[1]=0.3 is not (0.3<0.8).
        // t=2: row[0]=0.4 not (0.5>0.4); row[1]=0.8 is peak (0.3<0.8>0.1).
        // t=3: row[0]=0.6 is peak (0.4<0.6>0.0); row[1]=0.1 not.
        let peaks = argrelmax_axis0(&grid);
        assert_eq!(peaks, vec![(1, 0), (2, 1), (3, 0)]);
    }

    #[test]
    fn inferred_onsets_takes_max_of_onset_and_scaled_diff() {
        // 3 note_grid, 1 freq. note_grid jump 0 -> 0.5 -> 1.0 so frame_diff[2] = 0.5.
        // onset_grid are flat 0.0, so the inferred path should dominate at t=2.
        let note_grid = vec![vec![0.0], vec![0.5], vec![1.0]];
        let onset_grid = vec![vec![0.0], vec![0.0], vec![0.0]];
        let combined = inferred_onsets(&onset_grid, &note_grid);
        // max_onset is 0.0 -> scale = 0.0 -> combined collapses to onset_grid.
        assert_eq!(combined, vec![vec![0.0], vec![0.0], vec![0.0]]);
    }

    #[test]
    fn inferred_onsets_scales_diffs_to_onset_range() {
        // when onset has signal, the diff gets scaled up to its max.
        let note_grid = vec![vec![0.0], vec![0.0], vec![0.5], vec![1.0]];
        let onset_grid = vec![vec![0.0], vec![0.4], vec![0.0], vec![0.0]];
        let combined = inferred_onsets(&onset_grid, &note_grid);
        // t=2 frame_diff = min(0.5-0.0, 0.5-0.0) = 0.5 (scaled to 0.4)
        // t=3 frame_diff = min(1.0-0.5, 1.0-0.0) = 0.5 (scaled to 0.4)
        assert!((combined[1][0] - 0.4).abs() < 1e-6);
        assert!((combined[2][0] - 0.4).abs() < 1e-6);
        assert!((combined[3][0] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn midi_to_contour_bin_anchors() {
        // midi 21 (A0) is fmin; bin 0.
        assert!((midi_to_contour_bin(21) - 0.0).abs() < 1e-9);
        // one octave up (midi 33) = 12 semitones * 3 bins/semitone = 36 bins.
        assert!((midi_to_contour_bin(33) - 36.0).abs() < 1e-9);
    }

    #[test]
    fn gaussian_window_is_symmetric_unit_peak() {
        let w = gaussian_window(11, 2.0);
        // peak at the center.
        let max_idx = w
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 5);
        assert!((w[5] - 1.0).abs() < 1e-9);
        // symmetric.
        for i in 0..5 {
            assert!((w[i] - w[10 - i]).abs() < 1e-9);
        }
    }

    #[test]
    fn detect_notes_parity() -> Result<()> {
        let device = Device::Cpu;
        let fixture = load_fixture(&device);

        let notes = detect_notes(
            &fixture["full_contour"],
            &fixture["full_note"],
            &fixture["full_onset"],
            0.5,
            0.3,
            11,
        )?;

        let expected_starts: Vec<i64> = fixture["notes.start_frames"].to_vec1()?;
        let expected_ends: Vec<i64> = fixture["notes.end_frames"].to_vec1()?;
        let expected_midis: Vec<i64> = fixture["notes.midi"].to_vec1()?;
        let expected_bends: Vec<i64> = fixture["notes.bends_concat"].to_vec1()?;
        let expected_offsets: Vec<i64> = fixture["notes.bends_offsets"].to_vec1()?;

        assert_eq!(notes.len(), expected_starts.len(), "note count");

        for (i, note) in notes.iter().enumerate() {
            let start_frame =
                (note.points[0].time * SAMPLE_RATE as f32 / HOP as f32).round() as i64;
            let end_frame = start_frame + note.points.len() as i64;
            assert_eq!(start_frame, expected_starts[i], "note {i} start_frame");
            assert_eq!(end_frame, expected_ends[i], "note {i} end_frame");

            let bend_lo = expected_offsets[i] as usize;
            let bend_hi = expected_offsets[i + 1] as usize;
            let bends = &expected_bends[bend_lo..bend_hi];
            assert_eq!(note.points.len(), bends.len(), "note {i} bend count");
            let central_hz = midi_to_hz(expected_midis[i] as u8);
            for (point, &bend) in note.points.iter().zip(bends) {
                let cents = bend as f32 * (100.0 / CONTOURS_BINS_PER_SEMITONE as f32);
                let expected_hz = central_hz * 2f32.powf(cents / 1200.0);
                let diff = (point.frequency - expected_hz).abs();
                assert!(diff < 1e-3, "note {i} frequency drift {diff:.2e}");
            }
        }
        Ok(())
    }
}
