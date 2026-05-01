pub mod notes;

use clap::ValueEnum;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Capacity {
    Tiny,
    Small,
    Medium,
    Large,
    Full,
}

impl From<Capacity> for candle_crepe::Capacity {
    fn from(c: Capacity) -> Self {
        match c {
            Capacity::Tiny => Self::Tiny,
            Capacity::Small => Self::Small,
            Capacity::Medium => Self::Medium,
            Capacity::Large => Self::Large,
            Capacity::Full => Self::Full,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Decoder {
    Local,
    Viterbi,
}

impl From<Decoder> for candle_crepe::Decoder {
    fn from(d: Decoder) -> Self {
        match d {
            Decoder::Local => Self::LocalAverage,
            Decoder::Viterbi => Self::Viterbi,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum DeviceKind {
    #[default]
    Auto,
    Cpu,
    #[cfg(feature = "metal")]
    Metal,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl DeviceKind {
    pub fn into_device(self) -> candle_core::Result<candle_core::Device> {
        match self {
            DeviceKind::Auto => auto_select(),
            DeviceKind::Cpu => Ok(candle_core::Device::Cpu),
            #[cfg(feature = "metal")]
            DeviceKind::Metal => candle_core::Device::new_metal(0),
            #[cfg(feature = "cuda")]
            DeviceKind::Cuda => candle_core::Device::new_cuda(0),
        }
    }
}

fn auto_select() -> candle_core::Result<candle_core::Device> {
    #[cfg(feature = "cuda")]
    if let Ok(d) = candle_core::Device::new_cuda(0) {
        eprintln!("device: cuda");
        return Ok(d);
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = candle_core::Device::new_metal(0) {
        eprintln!("device: metal");
        return Ok(d);
    }
    Ok(candle_core::Device::Cpu)
}
