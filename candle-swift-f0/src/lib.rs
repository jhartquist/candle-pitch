mod frontend;
mod inference;
mod model;
mod weights;

#[cfg(test)]
mod test_helpers;

pub use inference::{
    FRAME_LENGTH, HOP_LENGTH, PredictionFrame, SAMPLE_RATE, STFT_PADDING, predict,
};
pub use model::{PITCH_BINS, SwiftF0};

#[cfg(feature = "hf-hub")]
pub const HUB_REPO_ID: &str = "jhartquist/swift-f0";

#[cfg(feature = "hf-hub")]
pub const HUB_WEIGHTS_FILENAME: &str = "swift-f0.safetensors";
