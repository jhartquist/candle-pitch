use std::path::PathBuf;

use candle_core::Device;
use candle_pitch::notes::{self, Note, PredictionFrame};
use candle_pitch::{Capacity, Decoder, DeviceKind};
use clap::{Parser, Subcommand};
use serde::Serialize;

use crate::audio::{self, AudioSource};
use crate::plot;

const TARGET_SAMPLE_RATE: u32 = 16_000;
const VOICING_THRESHOLD: f32 = 0.5;
const MAX_PITCH_JUMP_CENTS: f32 = 100.0;
const MIN_NOTE_DURATION_SECONDS: f32 = 0.05;

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

    /// Inference device. Auto picks cuda > metal > cpu among compiled-in backends.
    #[arg(long, value_enum, default_value_t = DeviceKind::default())]
    device: DeviceKind,

    #[command(subcommand)]
    model: Model,
}

#[derive(Subcommand)]
enum Model {
    /// CREPE: monophonic pitch detection (Kim et al., 2018).
    Crepe(CrepeArgs),
    /// Swift-F0: monophonic pitch detection (Nieradzik, 2024).
    SwiftF0(SwiftF0Args),
}

#[derive(Parser)]
struct CrepeArgs {
    /// Directory containing CREPE weights (one safetensors per capacity).
    #[arg(long, default_value = "candle-crepe/weights")]
    weights_dir: PathBuf,

    /// Model capacity.
    #[arg(long, value_enum, default_value = "full")]
    capacity: Capacity,

    /// Fundamental-frequency decoder.
    #[arg(long, value_enum, default_value = "viterbi")]
    decoder: Decoder,
}

#[derive(Parser)]
struct SwiftF0Args {
    /// Path to the Swift-F0 safetensors file.
    #[arg(long, default_value = "candle-swift-f0/weights/swift-f0.safetensors")]
    weights: PathBuf,
}

struct Inference {
    name: &'static str,
    capacity: Option<String>,
    decoder: Option<&'static str>,
    frames: Vec<PredictionFrame>,
}

#[derive(Serialize)]
struct Output {
    model: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    capacity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decoder: Option<&'static str>,
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
    let AudioSource {
        samples,
        sample_rate,
        channels,
        duration_seconds,
    } = audio::load_audio(&args.input)?;
    let samples_mono = audio::downmix(samples, channels as usize);
    let samples_mono_16k = audio::resample(samples_mono, sample_rate, TARGET_SAMPLE_RATE)?;

    let device = args.device.into_device()?;

    let inference = match args.model {
        Model::Crepe(crepe_args) => run_crepe(crepe_args, &samples_mono_16k, &device)?,
        Model::SwiftF0(swift_args) => run_swift_f0(swift_args, &samples_mono_16k, &device)?,
    };

    let notes = notes::segment(
        &inference.frames,
        VOICING_THRESHOLD,
        MAX_PITCH_JUMP_CENTS,
        MIN_NOTE_DURATION_SECONDS,
    );

    let output = Output {
        model: inference.name,
        capacity: inference.capacity,
        decoder: inference.decoder,
        audio: AudioInfo {
            path: args.input,
            sample_rate,
            channels,
            duration_seconds,
        },
        frames: inference
            .frames
            .iter()
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

fn run_crepe(args: CrepeArgs, audio: &[f32], device: &Device) -> Result<Inference, DynError> {
    let capacity = candle_crepe::Capacity::from(args.capacity);
    let weights_path = args.weights_dir.join(format!("{capacity}.safetensors"));
    let weights = std::fs::read(&weights_path)
        .map_err(|e| format!("read weights {}: {e}", weights_path.display()))?;
    let model = candle_crepe::Crepe::from_safetensors(&weights, device)?;
    let predictions = candle_crepe::predict(&model, audio, args.decoder.into())?;
    let decoder_name = match args.decoder {
        Decoder::Local => "local",
        Decoder::Viterbi => "viterbi",
    };
    Ok(Inference {
        name: "crepe",
        capacity: Some(capacity.to_string()),
        decoder: Some(decoder_name),
        frames: predictions.into_iter().map(Into::into).collect(),
    })
}

fn run_swift_f0(args: SwiftF0Args, audio: &[f32], device: &Device) -> Result<Inference, DynError> {
    let weights = std::fs::read(&args.weights)
        .map_err(|e| format!("read weights {}: {e}", args.weights.display()))?;
    let model = candle_swift_f0::SwiftF0::from_safetensors(&weights, device)?;
    let predictions = candle_swift_f0::predict(&model, audio)?;
    Ok(Inference {
        name: "swift-f0",
        capacity: None,
        decoder: None,
        frames: predictions.into_iter().map(Into::into).collect(),
    })
}
