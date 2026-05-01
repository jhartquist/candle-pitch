#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "basic-pitch[tf]",
#     "librosa>=0.10",
#     "nnAudio>=0.3.4",
#     "numpy<2",
#     "safetensors>=0.4",
#     "setuptools<80",
#     "torch>=2.1",
# ]
# ///
"""
PyTorch reimplementation of Basic Pitch, plus parity check and fixture dump.

Defines the model as Conv2d + BatchNorm2d (matching the safetensors layout
produced by export_safetensors.py), loads weights, runs windowed inference
with nnAudio's CQT2010v2 frontend, and verifies bit parity against the
upstream TF reference on the project's test audio.

Once parity holds, dumps a safetensors fixture under
tests/fixtures/basic_pitch.safetensors with:
  - full audio + full posteriorgrams (contour/note/onset)
  - one window's worth of every per-layer intermediate, including the
    raw CQT magnitudes — so the Rust crate can layer-by-layer reproduce
    the pipeline without re-running Python
  - the note events that upstream's polyphonic detector emits, with
    per-frame integer pitch bends in contour bins.

Usage:
    ./pytorch_parity.py <weights.safetensors>
"""

import math
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.numpy import save_file
from safetensors.torch import load_file


# audio / model constants — must match Spotify's ICASSP 2022 model exactly.
SR = 22050
HOP = 256
FMIN = 27.5
N_BINS = 309
BPO = 36
N_HARMONICS = 8
N_OUTPUT_FREQS = 264          # 88 piano keys * 3 contour bins/semitone
BINS_PER_SEMITONE = 3
BN_EPS = 0.001                # Keras/TF default

# windowing (verbatim from basic_pitch.inference.run_inference).
AUDIO_N_SAMPLES = 43844
N_OVERLAPPING_FRAMES = 30
OVERLAP_SAMPLES = N_OVERLAPPING_FRAMES * HOP
HOP_SAMPLES = AUDIO_N_SAMPLES - OVERLAP_SAMPLES
N_OLAP_TRIM = N_OVERLAPPING_FRAMES // 2  # frames trimmed each end per window
ANNOTATIONS_FPS = SR // HOP   # 86

# note-detector params (defaults from basic_pitch.predict).
ONSET_THRESH = 0.5
FRAME_THRESH = 0.3
MIN_NOTE_LEN_FRAMES = 11

PARITY_TOL = 1e-3

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIO_PATH = REPO_ROOT / "data" / "AMajSlow.wav"


# harmonic-stacking shifts in CQT bins. matches basic_pitch.nn.HarmonicStacking
# init logic: shift = round(BPO * log2(h)) for h in HARMONICS.
HARMONICS = [0.5, 1, 2, 3, 4, 5, 6, 7]
HARMONIC_SHIFTS = [
    0 if h == 1 else int(round(BPO * math.log2(float(h)))) for h in HARMONICS
]


def conv2d_same(x, weight, bias, stride=(1, 1)):
    """Keras-style SAME padding: out = ceil(in / stride). Stride (1, 3) makes
    PyTorch's `padding='same'` unusable, so we compute asymmetric padding by
    hand. Pad-left = floor(pad/2), pad-right = pad - pad-left."""
    ih, iw = x.shape[2], x.shape[3]
    kh, kw = weight.shape[2], weight.shape[3]
    sh, sw = stride
    oh = math.ceil(ih / sh)
    ow = math.ceil(iw / sw)
    pad_h = max(0, (oh - 1) * sh + kh - ih)
    pad_w = max(0, (ow - 1) * sw + kw - iw)
    if pad_h or pad_w:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride=stride)


def harmonic_stacking(x):
    """CQT (B, 1, T, 264) -> HCQT (B, 8, T, 264) by stacking 8 frequency-shifted
    copies along the channel dim. Negative shift = pad on the left and crop the
    right (lower harmonic); positive = the opposite (higher harmonic)."""
    channels = []
    for shift in HARMONIC_SHIFTS:
        if shift == 0:
            ch = x
        elif shift > 0:
            ch = F.pad(x[:, :, :, shift:], (0, shift))
        else:
            ch = F.pad(x[:, :, :, :shift], (-shift, 0))
        channels.append(ch)
    return torch.cat(channels, dim=1)[:, :, :, :N_OUTPUT_FREQS]


def normalized_log(cqt_mag):
    """Min-max normalized log-power. CQT magnitudes are squared (-> power)
    before taking log10; the resulting tensor is then min-shifted to >=0 and
    divided by its max so values land in [0, 1]."""
    power = cqt_mag**2
    log_power = 10.0 * torch.log10(power + 1e-10)
    log_power = log_power - log_power.min()
    lp_max = log_power.max()
    if lp_max > 0:
        log_power = log_power / lp_max
    return log_power


class BasicPitchModel(nn.Module):
    """ICASSP 2022 architecture: 16,864 params, three posteriorgram heads."""

    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(1, eps=BN_EPS)
        self.contour_conv1 = nn.Conv2d(N_HARMONICS, 8, (3, 39), bias=True)
        self.bn2 = nn.BatchNorm2d(8, eps=BN_EPS)
        self.contour_conv2 = nn.Conv2d(8, 1, (5, 5), bias=True)
        self.note_conv1 = nn.Conv2d(1, 32, (7, 7), bias=True)
        self.note_conv2 = nn.Conv2d(32, 1, (7, 3), bias=True)
        self.onset_conv1 = nn.Conv2d(N_HARMONICS, 32, (5, 5), bias=True)
        self.bn3 = nn.BatchNorm2d(32, eps=BN_EPS)
        self.onset_conv2 = nn.Conv2d(33, 1, (3, 3), bias=True)


def forward_with_activations(model, normalized_spec):
    """Forward pass returning every named intermediate. The activation names
    are exactly what the Rust per-layer parity tests will read out of the
    safetensors fixture."""
    acts: dict[str, torch.Tensor] = {"normalized_log": normalized_spec}

    x = model.bn1(normalized_spec)
    acts["after_bn1"] = x

    hcqt = harmonic_stacking(x)
    acts["hcqt"] = hcqt

    x_c = conv2d_same(hcqt, model.contour_conv1.weight, model.contour_conv1.bias)
    acts["contour_conv1_out"] = x_c
    x_c = model.bn2(x_c)
    acts["after_bn2"] = x_c
    x_c = F.relu(x_c)
    x_c = conv2d_same(x_c, model.contour_conv2.weight, model.contour_conv2.bias)
    contour = torch.sigmoid(x_c)
    acts["contour"] = contour

    x_n = conv2d_same(contour, model.note_conv1.weight, model.note_conv1.bias, stride=(1, 3))
    acts["note_conv1_out"] = x_n
    x_n = F.relu(x_n)
    x_n = conv2d_same(x_n, model.note_conv2.weight, model.note_conv2.bias)
    note = torch.sigmoid(x_n)
    acts["note"] = note

    x_o = conv2d_same(hcqt, model.onset_conv1.weight, model.onset_conv1.bias, stride=(1, 3))
    acts["onset_conv1_out"] = x_o
    x_o = model.bn3(x_o)
    acts["after_bn3"] = x_o
    x_o = F.relu(x_o)
    x_o = torch.cat([note, x_o], dim=1)
    x_o = conv2d_same(x_o, model.onset_conv2.weight, model.onset_conv2.bias)
    onset = torch.sigmoid(x_o)
    acts["onset"] = onset

    return acts


def load_weights_into(model: BasicPitchModel, weights_path: Path) -> None:
    state = load_file(str(weights_path))
    result = model.load_state_dict(state, strict=False)
    ignorable = {f"bn{i}.num_batches_tracked" for i in (1, 2, 3)}
    missing = set(result.missing_keys) - ignorable
    unexpected = [k for k in result.unexpected_keys if not k.endswith("num_batches_tracked")]
    assert not missing, f"missing weights: {sorted(missing)}"
    assert not unexpected, f"unexpected weights: {unexpected}"
    model.eval()


def make_cqt_layer():
    from nnAudio.features.cqt import CQT2010v2
    return CQT2010v2(
        sr=SR, hop_length=HOP, fmin=FMIN, n_bins=N_BINS,
        bins_per_octave=BPO, pad_mode="reflect", verbose=False,
    )


@torch.no_grad()
def cqt_magnitude(cqt, audio):
    """Returns (n_frames, N_BINS) f32. nnAudio yields (B, F, T); we transpose."""
    x = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
    mag = cqt(x)                               # (1, N_BINS, n_frames)
    return mag[0].T.contiguous()               # (n_frames, N_BINS)


def window_audio(audio):
    """Prepend OVERLAP_SAMPLES//2 zeros, then slide AUDIO_N_SAMPLES windows
    at HOP_SAMPLES. Right-pad the final window with zeros if needed."""
    padded = np.concatenate(
        [np.zeros(OVERLAP_SAMPLES // 2, dtype=np.float32), audio.astype(np.float32)]
    )
    windows = []
    for start in range(0, len(padded), HOP_SAMPLES):
        win = padded[start : start + AUDIO_N_SAMPLES]
        if len(win) < AUDIO_N_SAMPLES:
            win = np.pad(win, (0, AUDIO_N_SAMPLES - len(win)))
        windows.append(win)
    return windows


def unwrap_window_outputs(per_window, audio_len):
    """Trim N_OLAP_TRIM frames from each end of each window's output, then
    concatenate and clip to floor(audio_len * fps / sr) frames. Matches
    basic_pitch.inference.unwrap_output."""
    stacked = np.stack(per_window)  # (n_windows, T_w, n_freqs)
    if N_OLAP_TRIM > 0:
        stacked = stacked[:, N_OLAP_TRIM:-N_OLAP_TRIM, :]
    flat = stacked.reshape(-1, stacked.shape[-1])
    n_expected = int(np.floor(audio_len * ANNOTATIONS_FPS / SR))
    return flat[:n_expected]


@torch.no_grad()
def run_pytorch_windowed(model, cqt, audio):
    """Returns (full_outputs, first_window_acts) where first_window_acts is
    every named intermediate from window 0 — used to populate the per-layer
    fixture slice."""
    windows = window_audio(audio)
    note_wins, onset_wins, contour_wins = [], [], []
    first_acts = None
    for i, win in enumerate(windows):
        mag = cqt_magnitude(cqt, win)                              # (T_w, N_BINS)
        spec = normalized_log(mag.unsqueeze(0).unsqueeze(0))       # (1,1,T_w,N_BINS)
        acts = forward_with_activations(model, spec)
        # remove the leading (B,C) singleton dims for the fixture; matches
        # the Rust convention of (T_w, *) tensors.
        if i == 0:
            first_acts = {k: v.cpu().numpy() for k, v in acts.items()}
            first_acts["cqt_mag"] = mag.cpu().numpy()
            first_acts["audio_window"] = win.astype(np.float32)
        note_wins.append(acts["note"][0, 0].cpu().numpy())
        onset_wins.append(acts["onset"][0, 0].cpu().numpy())
        contour_wins.append(acts["contour"][0, 0].cpu().numpy())

    full = {
        "note": unwrap_window_outputs(note_wins, len(audio)),
        "onset": unwrap_window_outputs(onset_wins, len(audio)),
        "contour": unwrap_window_outputs(contour_wins, len(audio)),
    }
    return full, first_acts


def run_tf(audio_path):
    """Upstream TF reference inference."""
    from contextlib import redirect_stdout
    from io import StringIO
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model as TFModel, run_inference

    model = TFModel(ICASSP_2022_MODEL_PATH)
    with redirect_stdout(StringIO()):
        out = run_inference(str(audio_path), model)
    return {k: np.asarray(v) for k, v in out.items()}


def detect_notes(full_outputs):
    """Run upstream's polyphonic note detector + pitch-bend extractor.

    Returns parallel arrays:
        starts/ends:  frame indices (int64)
        midis:        MIDI pitches (int64)
        amplitudes:   mean note posteriorgram amplitude (f32)
        bends_concat: flat int64 concat of every note's pitch_bends
        bends_offsets:cumulative starts; bends for note i live in
                      bends_concat[bends_offsets[i] .. bends_offsets[i+1]]

    Frame indices are pre-`model_frames_to_time` so the Rust crate can do
    its own integer-indexed reasoning, then convert to seconds at the end.
    """
    from basic_pitch.note_creation import (
        get_pitch_bends,
        output_to_notes_polyphonic,
    )

    raw = output_to_notes_polyphonic(
        full_outputs["note"],
        full_outputs["onset"],
        onset_thresh=ONSET_THRESH,
        frame_thresh=FRAME_THRESH,
        infer_onsets=True,
        min_note_len=MIN_NOTE_LEN_FRAMES,
        min_freq=None,
        max_freq=None,
        melodia_trick=True,
    )
    notes_with_bends = get_pitch_bends(full_outputs["contour"], raw)

    starts, ends, midis, amps = [], [], [], []
    bends_concat: list[int] = []
    bends_offsets: list[int] = [0]
    for s, e, p, a, bends in notes_with_bends:
        starts.append(s)
        ends.append(e)
        midis.append(p)
        amps.append(a)
        bends_concat.extend(int(b) for b in (bends or []))
        bends_offsets.append(len(bends_concat))

    return {
        "notes.start_frames": np.asarray(starts, dtype=np.int64),
        "notes.end_frames": np.asarray(ends, dtype=np.int64),
        "notes.midi": np.asarray(midis, dtype=np.int64),
        "notes.amplitude": np.asarray(amps, dtype=np.float32),
        "notes.bends_concat": np.asarray(bends_concat, dtype=np.int64),
        "notes.bends_offsets": np.asarray(bends_offsets, dtype=np.int64),
    }


def dump_fixture(audio, full_outputs, first_acts, notes, out_path):
    """Layer-by-layer parity slice (`window_*`) plus full-clip outputs."""
    fixture: dict[str, np.ndarray] = {
        "audio_22k": np.ascontiguousarray(audio.astype(np.float32)),
        "window_audio": np.ascontiguousarray(first_acts["audio_window"]),
        "window_cqt_mag": np.ascontiguousarray(first_acts["cqt_mag"].astype(np.float32)),
    }
    # window-scope intermediates — strip leading (1, 1) for 2D layers,
    # leading (1,) for 4D layers. Anything that comes out 4D stays as
    # (C, T, F); 5D stays as (C, T, F).
    layer_keys = (
        "normalized_log", "after_bn1", "hcqt",
        "contour_conv1_out", "after_bn2", "contour",
        "note_conv1_out", "note",
        "onset_conv1_out", "after_bn3", "onset",
    )
    for k in layer_keys:
        arr = first_acts[k]
        # all are (1, C, T, F); squeeze the batch
        assert arr.shape[0] == 1, f"{k}: unexpected batch {arr.shape}"
        fixture[f"window_{k}"] = np.ascontiguousarray(arr[0].astype(np.float32))

    fixture["full_contour"] = np.ascontiguousarray(full_outputs["contour"].astype(np.float32))
    fixture["full_note"] = np.ascontiguousarray(full_outputs["note"].astype(np.float32))
    fixture["full_onset"] = np.ascontiguousarray(full_outputs["onset"].astype(np.float32))

    fixture.update(notes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(fixture, str(out_path))


def report(label, ours, theirs):
    diff = float(np.abs(ours - theirs).max())
    print(f"  {label:30s} max abs diff {diff:.2e}")
    return diff


def verify(weights_path):
    audio, _ = librosa.load(str(AUDIO_PATH), sr=SR, mono=True)
    audio = audio.astype(np.float32, copy=False)

    model = BasicPitchModel()
    load_weights_into(model, weights_path)
    cqt = make_cqt_layer().eval()

    full_pt, first_acts = run_pytorch_windowed(model, cqt, audio)
    full_tf = run_tf(AUDIO_PATH)

    print(
        f"basic-pitch: PyTorch parity on {AUDIO_PATH.name} "
        f"({len(audio)} samples, {full_pt['note'].shape[0]} frames)\n"
    )

    worst = 0.0
    for key in ("note", "onset", "contour"):
        a = full_pt[key]
        b = full_tf[key]
        # tf returns (1, T, F); strip leading singletons
        while b.ndim > a.ndim and b.shape[0] == 1:
            b = b[0]
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]
        worst = max(worst, report(f"{key:7s}    vs tf", a, b))
    assert worst < PARITY_TOL, f"TF/PT disagree by {worst:.2e} > {PARITY_TOL:.0e}"

    print(f"\nparity verified ({worst:.2e} < {PARITY_TOL:.0e})")

    notes = detect_notes(full_pt)
    n_notes = len(notes["notes.start_frames"])
    print(f"\ndetected {n_notes} notes via upstream polyphonic detector + pitch bends")

    out_path = REPO_ROOT / "candle-basic-pitch" / "tests" / "fixtures" / "basic_pitch.safetensors"
    dump_fixture(audio, full_pt, first_acts, notes, out_path)
    rel = out_path.relative_to(REPO_ROOT)
    print(
        f"\nwrote fixture: {rel}\n"
        f"  audio_22k         {audio.shape}\n"
        f"  window_*          one window (first) including raw audio, cqt_mag, and per-layer intermediates\n"
        f"  full_*            {full_pt['note'].shape[0]} frames (contour/note/onset)\n"
        f"  notes.*           {n_notes} note events with pitch bends ({len(notes['notes.bends_concat'])} bend samples)"
    )


def main():
    if len(sys.argv) != 2:
        prog = Path(sys.argv[0]).name
        print(f"usage: {prog} <weights.safetensors>", file=sys.stderr)
        sys.exit(2)
    verify(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
