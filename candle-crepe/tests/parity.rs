use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};
use candle_crepe::{Capacity, Crepe, Decoder, HOP_LENGTH, PredictionFrame, SAMPLE_RATE, predict};

const TOLERANCE: f32 = 1e-4;

macro_rules! parity_for_capacity {
    ($mod_name:ident, $capacity:expr) => {
        mod $mod_name {
            use super::*;

            const CAPACITY: Capacity = $capacity;

            #[test]
            fn capacity_inferred_from_weights() -> Result<()> {
                let device = Device::Cpu;
                let model = Crepe::from_safetensors(&load_weights(CAPACITY), &device)?;
                assert_eq!(model.capacity(), CAPACITY);
                Ok(())
            }

            #[test]
            fn salience_parity() -> Result<()> {
                let device = Device::Cpu;
                let expected = load_fixture(CAPACITY, &device);
                let model = Crepe::from_safetensors(&load_weights(CAPACITY), &device)?;
                let audio: Vec<f32> = expected["audio_16k"].to_vec1()?;
                let salience = model.salience(&audio)?;
                let diff = max_abs_diff(&salience, &expected["salience"])?;
                assert!(diff < TOLERANCE, "salience diff {diff:.2e}");
                Ok(())
            }

            #[test]
            fn predict_local_average_parity() -> Result<()> {
                let device = Device::Cpu;
                let expected = load_fixture(CAPACITY, &device);
                let model = Crepe::from_safetensors(&load_weights(CAPACITY), &device)?;
                let audio: Vec<f32> = expected["audio_16k"].to_vec1()?;
                let predictions = predict(&model, &audio, Decoder::LocalAverage)?;
                assert_predictions(&predictions, &expected, "frequency_local")
            }

            #[test]
            fn predict_viterbi_parity() -> Result<()> {
                let device = Device::Cpu;
                let expected = load_fixture(CAPACITY, &device);
                let model = Crepe::from_safetensors(&load_weights(CAPACITY), &device)?;
                let audio: Vec<f32> = expected["audio_16k"].to_vec1()?;
                let predictions = predict(&model, &audio, Decoder::Viterbi)?;
                assert_predictions(&predictions, &expected, "frequency_viterbi")
            }
        }
    };
}

parity_for_capacity!(tiny, Capacity::Tiny);
parity_for_capacity!(small, Capacity::Small);
parity_for_capacity!(medium, Capacity::Medium);
parity_for_capacity!(large, Capacity::Large);
parity_for_capacity!(full, Capacity::Full);

#[cfg(feature = "hf-hub")]
#[test]
#[ignore = "downloads weights from huggingface.co"]
fn from_hub_loads_tiny() -> Result<()> {
    let device = Device::Cpu;
    let model = Crepe::from_hub(Capacity::Tiny, &device)?;
    assert_eq!(model.capacity(), Capacity::Tiny);
    Ok(())
}

fn load_weights(capacity: Capacity) -> Vec<u8> {
    let path = format!(
        "{}/weights/{capacity}.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn load_fixture(capacity: Capacity, device: &Device) -> HashMap<String, Tensor> {
    let path = format!(
        "{}/tests/fixtures/{capacity}.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    candle_core::safetensors::load_buffer(&bytes, device).expect("load fixture")
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
}

fn max_cents_diff(ours: &[f32], theirs: &[f32]) -> f32 {
    assert_eq!(ours.len(), theirs.len());
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| {
            if *a > 0.0 && *b > 0.0 {
                1200.0 * (a / b).log2().abs()
            } else if *a == 0.0 && *b == 0.0 {
                0.0
            } else {
                f32::INFINITY
            }
        })
        .fold(0.0_f32, f32::max)
}

fn max_abs_slice(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn assert_predictions(
    predictions: &[PredictionFrame],
    expected: &HashMap<String, Tensor>,
    frequency_key: &str,
) -> Result<()> {
    let expected_frequencies: Vec<f32> = expected[frequency_key].to_vec1()?;
    let expected_confidences: Vec<f32> = expected["confidence"].to_vec1()?;
    assert_eq!(predictions.len(), expected_frequencies.len());

    let frequencies: Vec<f32> = predictions.iter().map(|p| p.frequency_hz).collect();
    let confidences: Vec<f32> = predictions.iter().map(|p| p.confidence).collect();

    let cents_diff = max_cents_diff(&frequencies, &expected_frequencies);
    assert!(cents_diff < 1e-2, "freq cents diff {cents_diff:.2e}");

    let conf_diff = max_abs_slice(&confidences, &expected_confidences);
    assert!(conf_diff < TOLERANCE, "conf diff {conf_diff:.2e}");

    let time_step = HOP_LENGTH as f32 / SAMPLE_RATE as f32;
    let max_time_diff = predictions
        .iter()
        .enumerate()
        .map(|(i, p)| (p.time_seconds - i as f32 * time_step).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_time_diff < 1e-6, "time diff {max_time_diff:.2e}");
    Ok(())
}
