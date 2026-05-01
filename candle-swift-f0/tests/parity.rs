use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};
use candle_swift_f0::{HOP_LENGTH, PredictionFrame, SAMPLE_RATE, SwiftF0, predict};

const TOLERANCE: f32 = 1e-4;

#[test]
#[ignore]
fn from_safetensors_loads() -> Result<()> {
    let device = Device::Cpu;
    let _model = SwiftF0::from_safetensors(&load_weights(), &device)?;
    Ok(())
}

#[test]
#[ignore]
fn forward_parity() -> Result<()> {
    let device = Device::Cpu;
    let expected = load_fixture(&device);
    let model = SwiftF0::from_safetensors(&load_weights(), &device)?;
    let audio: Vec<f32> = expected["audio_16k"].to_vec1()?;
    let slice_start = expected["slice_start"].to_scalar::<i64>()? as usize;

    let logits = model.forward(&audio)?.narrow(1, slice_start, 16)?;
    let diff = max_abs_diff(&logits, &expected["freq_projection"])?;
    assert!(diff < TOLERANCE, "logits diff {diff:.2e}");
    Ok(())
}

#[test]
#[ignore]
fn predict_parity() -> Result<()> {
    let device = Device::Cpu;
    let expected = load_fixture(&device);
    let model = SwiftF0::from_safetensors(&load_weights(), &device)?;
    let audio: Vec<f32> = expected["audio_16k"].to_vec1()?;
    let predictions = predict(&model, &audio)?;
    assert_predictions(&predictions, &expected)
}

fn load_weights() -> Vec<u8> {
    let path = format!(
        "{}/weights/swift-f0.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn load_fixture(device: &Device) -> HashMap<String, Tensor> {
    let path = format!(
        "{}/tests/fixtures/swift-f0.safetensors",
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
) -> Result<()> {
    let expected_pitch: Vec<f32> = expected["pitch_hz"].to_vec1()?;
    let expected_conf: Vec<f32> = expected["confidence"].to_vec1()?;
    assert_eq!(predictions.len(), expected_pitch.len());

    let pitch: Vec<f32> = predictions.iter().map(|p| p.pitch_hz).collect();
    let conf: Vec<f32> = predictions.iter().map(|p| p.confidence).collect();

    let cents_diff = max_cents_diff(&pitch, &expected_pitch);
    assert!(cents_diff < TOLERANCE, "pitch cents diff {cents_diff:.2e}");

    let conf_diff = max_abs_slice(&conf, &expected_conf);
    assert!(conf_diff < TOLERANCE, "conf diff {conf_diff:.2e}");

    let time_step = HOP_LENGTH as f32 / SAMPLE_RATE as f32;
    let max_time_diff = predictions
        .iter()
        .enumerate()
        .map(|(i, p)| (p.time_seconds - i as f32 * time_step).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_time_diff < 1e-3, "time diff {max_time_diff:.2e}");
    Ok(())
}
