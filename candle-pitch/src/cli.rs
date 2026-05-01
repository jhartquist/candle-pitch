use std::path::PathBuf;

use candle_crepe::{Crepe, predict};
use candle_pitch::notes::{self, Note};
use candle_pitch::{Capacity, Decoder, DeviceKind};
use clap::Parser;
use serde::Serialize;

use crate::audio::{self, AudioSource};
use crate::plot;

const TARGET_SAMPLE_RATE: u32 = 16_000;

type DynError = Box<dyn std::error::Error>;

#[derive(Parser)]
#[command(name = "candle-pitch", version, about)]
pub struct Args {
    /// Input audio file. WAV, MP3, FLAC, OGG, AAC, M4A.
    input: PathBuf,

    /// Output path. Writes JSON to stdout if omitted.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Plot path. Renders notes to PNG (or SVG if path ends in .svg).
    #[arg(long)]
    plot: Option<PathBuf>,

    /// Directory containing model weights.
    #[arg(long, default_value = "candle-crepe/weights")]
    weights_dir: PathBuf,

    /// CREPE model capacity.
    #[arg(long, value_enum, default_value = "full")]
    capacity: Capacity,

    /// Fundamental-frequency decoder.
    #[arg(long, value_enum, default_value = "viterbi")]
    decoder: Decoder,

    /// Inference device. Auto picks cuda > metal > cpu among compiled-in backends.
    #[arg(long, value_enum, default_value_t = DeviceKind::default())]
    device: DeviceKind,
}

#[derive(Serialize)]
struct Output {
    model: &'static str,
    capacity: String,
    decoder: &'static str,
    audio: AudioInfo,
    frames: Vec<Frame>,
    notes: Vec<Note>,
}

#[derive(Serialize)]
struct AudioInfo {
    path: PathBuf,
    sample_rate: u32,
    channels: u16,
    duration_seconds: f32,
}

#[derive(Serialize)]
struct Frame {
    time: f32,
    f0: f32,
    confidence: f32,
}

pub fn run(args: Args) -> Result<(), DynError> {
    // load & preprocess audio
    let AudioSource {
        samples,
        sample_rate,
        channels,
        duration_seconds,
    } = audio::load_audio(&args.input)?;
    let samples_mono = audio::downmix(samples, channels as usize);
    let samples_mono_16k = audio::resample(samples_mono, sample_rate, TARGET_SAMPLE_RATE)?;

    // load model & weights
    let crepe_capacity = candle_crepe::Capacity::from(args.capacity);
    let weights_path = args
        .weights_dir
        .join(format!("{crepe_capacity}.safetensors"));
    let weights = std::fs::read(&weights_path)
        .map_err(|e| format!("read weights {}: {e}", weights_path.display()))?;
    let device = args.device.into_device()?;
    let model = Crepe::from_safetensors(&weights, &device)?;

    // run inference
    let predictions = predict(&model, &samples_mono_16k, args.decoder.into())?;
    // 0.5 voicing threshold, 100 cents (1 semitone) max pitch jump, 50ms min note duration
    let notes = notes::segment(&predictions, 0.5, 100.0, 0.05);

    // assemble & return output
    let output = Output {
        model: "crepe",
        capacity: crepe_capacity.to_string(),
        decoder: match args.decoder {
            Decoder::Local => "local",
            Decoder::Viterbi => "viterbi",
        },
        audio: AudioInfo {
            path: args.input,
            sample_rate,
            channels,
            duration_seconds,
        },
        frames: predictions
            .into_iter()
            .map(|p| Frame {
                time: p.time_seconds,
                f0: p.frequency_hz,
                confidence: p.confidence,
            })
            .collect(),
        notes,
    };
    if let Some(plot_path) = &args.plot {
        plot::render_notes(&output.notes, plot_path)?;
    }

    let json = serde_json::to_string_pretty(&output)?;
    match args.output {
        Some(path) => std::fs::write(path, json)?,
        None => println!("{json}"),
    }
    Ok(())
}
