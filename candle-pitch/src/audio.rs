use std::fs::File;
use std::path::Path;

use rubato::{Fft, FixedSync, Resampler, audioadapter_buffers::direct::InterleavedSlice};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

type DynError = Box<dyn std::error::Error>;

pub struct AudioSource {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_seconds: f32,
}

pub fn load_audio(path: &Path) -> Result<AudioSource, DynError> {
    let mss = MediaSourceStream::new(Box::new(File::open(path)?), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("no decodable audio track")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or("unknown sample rate")?;
    let channels = track
        .codec_params
        .channels
        .ok_or("unknown channel layout")?
        .count() as u16;

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut samples: Vec<f32> = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet)?;
        let buf = buf.get_or_insert_with(|| {
            SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec())
        });
        buf.copy_interleaved_ref(decoded);
        samples.extend_from_slice(buf.samples());
    }

    let duration_seconds = (samples.len() / channels as usize) as f32 / sample_rate as f32;
    Ok(AudioSource {
        samples,
        sample_rate,
        channels,
        duration_seconds,
    })
}

pub fn downmix(samples: Vec<f32>, n_channels: usize) -> Vec<f32> {
    if n_channels == 1 {
        return samples;
    }
    let scale = 1.0 / n_channels as f32;
    samples
        .chunks_exact(n_channels)
        .map(|chunk| chunk.iter().sum::<f32>() * scale)
        .collect()
}

pub fn resample(input: Vec<f32>, source_rate: u32, target_rate: u32) -> Result<Vec<f32>, DynError> {
    if source_rate == target_rate {
        return Ok(input);
    }
    let mut resampler = Fft::<f32>::new(
        source_rate as usize,
        target_rate as usize,
        1024,
        2,
        1,
        FixedSync::Both,
    )?;
    let n_in = input.len();
    let n_out_capacity = resampler.process_all_needed_output_len(n_in);
    let mut output = vec![0f32; n_out_capacity];
    let in_adapter = InterleavedSlice::new(&input, 1, n_in)?;
    let mut out_adapter = InterleavedSlice::new_mut(&mut output, 1, n_out_capacity)?;
    let (_, n_out) =
        resampler.process_all_into_buffer(&in_adapter, &mut out_adapter, n_in, None)?;
    output.truncate(n_out);
    Ok(output)
}
