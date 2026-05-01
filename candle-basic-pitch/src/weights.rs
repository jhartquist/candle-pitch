use candle_core::{Device, Result};
use candle_nn::VarBuilder;

pub(crate) fn load_safetensors<'a>(bytes: &'a [u8], device: &Device) -> Result<VarBuilder<'a>> {
    todo!()
}
