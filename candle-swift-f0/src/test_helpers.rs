use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};

pub(crate) fn load_fixture(device: &Device) -> HashMap<String, Tensor> {
    let path = format!(
        "{}/tests/fixtures/swift-f0.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    candle_core::safetensors::load_buffer(&bytes, device).expect("load fixture")
}

pub(crate) fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
}

pub(crate) fn max_cents_diff(ours: &[f32], theirs: &[f32]) -> f32 {
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
