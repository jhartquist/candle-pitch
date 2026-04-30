use std::path::PathBuf;

use candle_pitch::{Capacity, Decoder};
use clap::Parser;

#[derive(Parser)]
#[command(name = "candle-pitch", version, about)]
struct Args {
    /// Input audio file. WAV, 16 kHz mono.
    input: PathBuf,

    /// Output path. Writes JSON to stdout if omitted.
    #[arg(long)]
    output: Option<PathBuf>,

    /// CREPE model capacity.
    #[arg(long, value_enum, default_value = "full")]
    capacity: Capacity,

    /// Fundamental-frequency decoder.
    #[arg(long, value_enum, default_value = "viterbi")]
    decoder: Decoder,
}

fn main() {
    let _args = Args::parse();
    todo!("pipeline lands in next commit")
}
