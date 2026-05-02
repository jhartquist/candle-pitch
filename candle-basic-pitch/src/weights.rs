use candle_core::{DType, Device, Result};
use candle_nn::VarBuilder;

pub(crate) fn load_safetensors<'a>(bytes: &'a [u8], device: &Device) -> Result<VarBuilder<'a>> {
    VarBuilder::from_slice_safetensors(bytes, DType::F32, device)
}
