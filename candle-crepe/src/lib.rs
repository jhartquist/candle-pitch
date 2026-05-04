mod inference;
mod model;
mod viterbi;
mod weights;

#[cfg(test)]
mod test_helpers;

pub use inference::{Decoder, FRAME_LENGTH, HOP_LENGTH, PredictionFrame, SAMPLE_RATE, predict};
pub use model::{Capacity, Crepe, PITCH_BINS};

#[cfg(feature = "hf-hub")]
pub const HUB_REPO_ID: &str = "jhartquist/crepe";
