use std::collections::HashMap;

use candle_basic_pitch::{BasicPitch, Frontend, Note, predict, run};
use candle_core::{Device, Result, Tensor};

const TOLERANCE: f32 = 1e-3;

#[test]
fn from_safetensors_loads() -> Result<()> {
    let device = Device::Cpu;
    let _frontend = Frontend::from_safetensors(&load_cqt_weights(), &device)?;
    let _model = BasicPitch::from_safetensors(&load_weights(), &device)?;
    Ok(())
}

#[test]
fn run_parity() -> Result<()> {
    let device = Device::Cpu;
    let expected = load_fixture(&device);
    let frontend = Frontend::from_safetensors(&load_cqt_weights(), &device)?;
    let model = BasicPitch::from_safetensors(&load_weights(), &device)?;
    let audio: Vec<f32> = expected["audio_22k"].to_vec1()?;

    let heads = run(&model, &frontend, &audio)?;
    let contour_diff = max_abs_diff(&heads.contour, &expected["full_contour"])?;
    let note_diff = max_abs_diff(&heads.note, &expected["full_note"])?;
    let onset_diff = max_abs_diff(&heads.onset, &expected["full_onset"])?;
    assert!(contour_diff < TOLERANCE, "contour diff {contour_diff:.2e}");
    assert!(note_diff < TOLERANCE, "note diff {note_diff:.2e}");
    assert!(onset_diff < TOLERANCE, "onset diff {onset_diff:.2e}");
    Ok(())
}

#[test]
#[ignore]
fn predict_parity() -> Result<()> {
    let device = Device::Cpu;
    let expected = load_fixture(&device);
    let frontend = Frontend::from_safetensors(&load_cqt_weights(), &device)?;
    let model = BasicPitch::from_safetensors(&load_weights(), &device)?;
    let audio: Vec<f32> = expected["audio_22k"].to_vec1()?;

    let notes = predict(&model, &frontend, &audio)?;
    assert_notes(&notes, &expected)
}

fn load_weights() -> Vec<u8> {
    let path = format!(
        "{}/weights/basic_pitch.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn load_cqt_weights() -> Vec<u8> {
    let path = format!("{}/weights/cqt.safetensors", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn load_fixture(device: &Device) -> HashMap<String, Tensor> {
    let path = format!(
        "{}/tests/fixtures/basic_pitch.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    candle_core::safetensors::load_buffer(&bytes, device).expect("load fixture")
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
}

fn assert_notes(notes: &[Note], expected: &HashMap<String, Tensor>) -> Result<()> {
    let starts: Vec<i64> = expected["notes.start_frames"].to_vec1()?;
    let ends: Vec<i64> = expected["notes.end_frames"].to_vec1()?;
    let midis: Vec<i64> = expected["notes.midi"].to_vec1()?;
    assert_eq!(notes.len(), starts.len(), "note count mismatch");
    // per-note frame/pitch checks land alongside the postprocess implementation.
    let _ = (ends, midis);
    Ok(())
}
