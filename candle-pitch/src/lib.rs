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
