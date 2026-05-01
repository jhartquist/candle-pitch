use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};

pub(crate) fn load_weights() -> Vec<u8> {
    let path = format!(
        "{}/weights/basic_pitch.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

pub(crate) fn load_cqt_weights() -> Vec<u8> {
    let path = format!("{}/weights/cqt.safetensors", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

pub(crate) fn load_fixture(device: &Device) -> HashMap<String, Tensor> {
    let path = format!(
        "{}/tests/fixtures/basic_pitch.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    candle_core::safetensors::load_buffer(&bytes, device).expect("load fixture")
}

pub(crate) fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
}
