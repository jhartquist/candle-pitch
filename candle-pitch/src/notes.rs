use candle_crepe::PredictionFrame;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Note {
    pub points: Vec<NotePoint>,
}

#[derive(Clone, Copy, Debug, Serialize)]
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

pub fn segment(
    predictions: &[PredictionFrame],
    voicing_threshold: f32,
    max_pitch_jump_cents: f32,
    min_duration_seconds: f32,
) -> Vec<Note> {
    let mut notes: Vec<Note> = Vec::new();
    let mut current: Option<Note> = None;
    let keep = |note: &Note| note.end() - note.start() >= min_duration_seconds;

    for frame in predictions {
        // unvoiced frames end the current note
        if frame.confidence < voicing_threshold {
            if let Some(note) = current.take()
                && keep(&note)
            {
                notes.push(note);
            }
            continue;
        }

        let point = NotePoint {
            time: frame.time_seconds,
            frequency: frame.frequency_hz,
            confidence: frame.confidence,
        };

        // a big pitch jump also ends the current note and starts a new one
        match &mut current {
            None => {
                current = Some(Note {
                    points: vec![point],
                })
            }
            Some(note) => {
                let last = note.points.last().unwrap();
                let cents = 1200.0 * (point.frequency / last.frequency).log2().abs();
                if cents > max_pitch_jump_cents {
                    let finished = current.take().unwrap();
                    if keep(&finished) {
                        notes.push(finished);
                    }
                    current = Some(Note {
                        points: vec![point],
                    });
                } else {
                    note.points.push(point);
                }
            }
        }
    }

    // flush trailing note
    if let Some(note) = current
        && keep(&note)
    {
        notes.push(note);
    }
    notes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(t: f32, frequency: f32, c: f32) -> PredictionFrame {
        PredictionFrame {
            time_seconds: t,
            frequency_hz: frequency,
            confidence: c,
        }
    }

    #[test]
    fn empty_input() {
        assert!(segment(&[], 0.5, 100.0, 0.0).is_empty());
    }

    #[test]
    fn all_unvoiced() {
        let frames = vec![frame(0.00, 440.0, 0.1), frame(0.01, 440.0, 0.2)];
        assert!(segment(&frames, 0.5, 100.0, 0.0).is_empty());
    }

    #[test]
    fn single_continuous_note() {
        let frames = vec![
            frame(0.00, 440.0, 0.9),
            frame(0.01, 441.0, 0.9),
            frame(0.02, 442.0, 0.9),
        ];
        let notes = segment(&frames, 0.5, 100.0, 0.0);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].points.len(), 3);
        assert_eq!(notes[0].start(), 0.00);
        assert_eq!(notes[0].end(), 0.02);
    }

    #[test]
    fn split_on_unvoiced() {
        let frames = vec![
            frame(0.00, 440.0, 0.9),
            frame(0.01, 440.0, 0.1),
            frame(0.02, 440.0, 0.9),
        ];
        let notes = segment(&frames, 0.5, 100.0, 0.0);
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].points.len(), 1);
        assert_eq!(notes[1].points.len(), 1);
    }

    #[test]
    fn split_on_pitch_jump() {
        // 440 Hz -> 880 Hz is 1200 cents, well above the 100 cent threshold
        let frames = vec![
            frame(0.00, 440.0, 0.9),
            frame(0.01, 440.0, 0.9),
            frame(0.02, 880.0, 0.9),
            frame(0.03, 880.0, 0.9),
        ];
        let notes = segment(&frames, 0.5, 100.0, 0.0);
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].points.len(), 2);
        assert_eq!(notes[1].points.len(), 2);
    }

    #[test]
    fn drops_notes_below_min_duration() {
        // two voiced clusters: a short one (20ms) followed by a long one (50ms),
        // separated by an unvoiced frame. min_duration of 30ms keeps only the long one.
        let frames = vec![
            frame(0.00, 440.0, 0.9),
            frame(0.01, 440.0, 0.9),
            frame(0.02, 440.0, 0.9),
            frame(0.03, 440.0, 0.1),
            frame(0.04, 880.0, 0.9),
            frame(0.05, 880.0, 0.9),
            frame(0.06, 880.0, 0.9),
            frame(0.07, 880.0, 0.9),
            frame(0.08, 880.0, 0.9),
            frame(0.09, 880.0, 0.9),
        ];
        let notes = segment(&frames, 0.5, 100.0, 0.03);
        assert_eq!(notes.len(), 1);
        assert!((notes[0].start() - 0.04).abs() < 1e-6);
    }

    #[test]
    fn small_pitch_drift_stays_one_note() {
        // ~50 cent drift over 4 frames, all below 100 cent jump threshold
        let frames = vec![
            frame(0.00, 440.0, 0.9),
            frame(0.01, 445.0, 0.9),
            frame(0.02, 450.0, 0.9),
            frame(0.03, 455.0, 0.9),
        ];
        let notes = segment(&frames, 0.5, 100.0, 0.0);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].points.len(), 4);
    }
}
