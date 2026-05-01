use candle_core::{Result, Tensor};

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

// run the upstream polyphonic note detector over assembled posteriorgrams,
// then attach per-frame pitch bends and convert to time-domain Notes.
pub(crate) fn detect_notes(
    contour: &Tensor,
    note: &Tensor,
    onset: &Tensor,
    onset_thresh: f32,
    frame_thresh: f32,
    min_note_len_frames: usize,
) -> Result<Vec<Note>> {
    todo!()
}
