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

const DEFAULT_CREPE_CAPACITY: Capacity = Capacity::Tiny;

type DynError = Box<dyn std::error::Error>;

#[derive(Parser)]
#[command(name = "candle-pitch", version, about)]
pub struct Args {
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
struct SharedArgs {
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
}

#[derive(Parser)]
struct CrepeArgs {
    #[command(flatten)]
    shared: SharedArgs,

    /// Model capacity. Defaults to tiny when --weights is not given;
    /// when --weights is given, the capacity is inferred from the file
    /// and this flag is checked for consistency.
    #[arg(long, value_enum)]
    capacity: Option<Capacity>,

    /// Fundamental-frequency decoder.
    #[arg(long, value_enum, default_value = "viterbi")]
    decoder: Decoder,

    /// Path to a CREPE safetensors file. If omitted, fetches the requested
    /// capacity from huggingface.co/jhartquist/crepe.
    #[arg(long)]
    weights: Option<PathBuf>,
}

#[derive(Parser)]
struct SwiftF0Args {
    #[command(flatten)]
    shared: SharedArgs,

    /// Path to the Swift-F0 safetensors file. If omitted, fetches from
    /// huggingface.co/jhartquist/swift-f0.
    #[arg(long)]
    weights: Option<PathBuf>,
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
    let shared = match &args.model {
        Model::Crepe(a) => &a.shared,
        Model::SwiftF0(a) => &a.shared,
    };
    let AudioSource {
        samples,
        sample_rate,
        channels,
        duration_seconds,
    } = audio::load_audio(&shared.input)?;
    let samples_mono = audio::downmix(samples, channels as usize);
    let samples_mono_16k = audio::resample(samples_mono, sample_rate, TARGET_SAMPLE_RATE)?;
    let device = shared.device.into_device()?;

    let (input, output_path, plot_path, inference) = match args.model {
        Model::Crepe(crepe_args) => {
            let SharedArgs {
                input,
                output,
                plot,
                ..
            } = crepe_args.shared;
            let inference = run_crepe(
                crepe_args.capacity,
                crepe_args.decoder,
                crepe_args.weights,
                &samples_mono_16k,
                &device,
            )?;
            (input, output, plot, inference)
        }
        Model::SwiftF0(swift_args) => {
            let SharedArgs {
                input,
                output,
                plot,
                ..
            } = swift_args.shared;
            let inference = run_swift_f0(swift_args.weights, &samples_mono_16k, &device)?;
            (input, output, plot, inference)
        }
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
            path: input,
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
    if let Some(path) = &plot_path {
        plot::render_notes(&output.notes, path)?;
    }

    let json = serde_json::to_string_pretty(&output)?;
    match output_path {
        Some(path) => std::fs::write(path, json)?,
        None => println!("{json}"),
    }
    Ok(())
}

fn run_crepe(
    requested_capacity: Option<Capacity>,
    decoder: Decoder,
    weights: Option<PathBuf>,
    audio: &[f32],
    device: &Device,
) -> Result<Inference, DynError> {
    let model = match weights {
        Some(path) => {
            let bytes = std::fs::read(&path)
                .map_err(|e| format!("read weights {}: {e}", path.display()))?;
            candle_crepe::Crepe::from_safetensors(&bytes, device)?
        }
        None => {
            let capacity = requested_capacity.unwrap_or(DEFAULT_CREPE_CAPACITY);
            candle_crepe::Crepe::from_hub(capacity.into(), device)?
        }
    };
    let loaded: Capacity = model.capacity().into();
    if let Some(requested) = requested_capacity
        && requested != loaded
    {
        return Err(format!(
            "--capacity {} does not match weights ({})",
            requested.name(),
            loaded.name()
        )
        .into());
    }

    let predictions = candle_crepe::predict(&model, audio, decoder.into())?;
    let decoder_name = match decoder {
        Decoder::Local => "local",
        Decoder::Viterbi => "viterbi",
    };
    Ok(Inference {
        name: "crepe",
        capacity: Some(loaded.name().to_string()),
        decoder: Some(decoder_name),
        frames: predictions.into_iter().map(Into::into).collect(),
    })
}

fn run_swift_f0(
    weights: Option<PathBuf>,
    audio: &[f32],
    device: &Device,
) -> Result<Inference, DynError> {
    let model = match weights {
        Some(path) => {
            let bytes = std::fs::read(&path)
                .map_err(|e| format!("read weights {}: {e}", path.display()))?;
            candle_swift_f0::SwiftF0::from_safetensors(&bytes, device)?
        }
        None => candle_swift_f0::SwiftF0::from_hub(device)?,
    };
    let predictions = candle_swift_f0::predict(&model, audio)?;
    Ok(Inference {
        name: "swift-f0",
        capacity: None,
        decoder: None,
        frames: predictions.into_iter().map(Into::into).collect(),
    })
}
