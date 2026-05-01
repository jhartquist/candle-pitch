#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["swift-f0", "librosa", "numpy", "onnxruntime", "safetensors", "torch"]
# ///
"""
PyTorch reimplementation of SwiftF0, plus parity check and fixture dump.

Defines the model as Conv2d + Conv1d (matching the safetensors layout
produced by export_safetensors.py), loads weights, and verifies bit
parity against the bundled ONNX on the project's test audio: pitch_hz
and confidence per frame.

Once parity holds, dumps a small safetensors fixture (resampled audio,
log-magnitude spectrogram, per-block activations, final pitch and
confidence) under tests/fixtures/swift-f0.safetensors. The Rust parity
tests consume this file and compare layer by layer.

Usage:
    ./pytorch_parity.py <weights.safetensors>
"""

import sys
from pathlib import Path

import librosa
import numpy as np
import onnxruntime
import swift_f0
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.numpy import save_file
from safetensors.torch import load_file


SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
STFT_PADDING = 384
K_MIN = 3
N_FREQ_BINS = 132
N_PITCH_BINS = 200
F_MIN = 46.875
F_MAX = 2093.75

CHANNELS = (8, 16, 32, 64, 1)
KERNEL = 5
PEAK_WIDTH = 9  # ±9 bins around argmax used for confidence + expectation

# size of the slice used for per-block intermediate fixtures. Per-block tensors
# scale with frame count, so we cap them at a small slice; the slice is picked
# in a high-confidence region so it covers actual notes, not silence/attack.
SLICE_FRAMES = 16

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIO_PATH = REPO_ROOT / "data" / "AMajSlow.wav"


class SwiftF0(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = [1, *CHANNELS[:-1]]
        for i, (ci, co) in enumerate(zip(in_channels, CHANNELS), start=1):
            self.add_module(f"conv{i}", nn.Conv2d(ci, co, KERNEL, padding=KERNEL // 2))
        self.freq_projection = nn.Conv1d(N_FREQ_BINS, N_PITCH_BINS, kernel_size=1)
        self.register_buffer("window", torch.hann_window(N_FFT))
        self.register_buffer("pitch_bin_centers", _build_pitch_bin_centers())

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_mag = self.frontend(audio)
        x = log_mag
        for i in range(1, len(CHANNELS) + 1):
            x = F.relu(getattr(self, f"conv{i}")(x))
        x = x.squeeze(1)
        logits = self.freq_projection(x).permute(0, 2, 1)
        return decode(logits, self.pitch_bin_centers)

    def frontend(self, audio: torch.Tensor) -> torch.Tensor:
        # The bundled ONNX uses constant (zero) padding, not reflect — even though
        # swift-f0's training/export.py says reflect. Source of truth is the ONNX.
        x = F.pad(audio, (STFT_PADDING, STFT_PADDING), mode="constant")
        x = torch.stft(
            x, n_fft=N_FFT, hop_length=HOP_LENGTH, window=self.window,
            center=False, return_complex=True,
        )
        mag = x.abs()[:, K_MIN : K_MIN + N_FREQ_BINS, :]
        return torch.log(mag + 1e-8).unsqueeze(1)


def _build_pitch_bin_centers() -> torch.Tensor:
    delta = float(np.log2(F_MAX / F_MIN)) / (N_PITCH_BINS - 1)
    bins = torch.arange(N_PITCH_BINS, dtype=torch.float32)
    return F_MIN * (2.0 ** (bins * delta))


def decode(logits: torch.Tensor, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pitch logits (B, T, N) -> (pitch_hz, confidence) of shape (B, T).
    Mass within ±PEAK_WIDTH bins of the per-frame argmax becomes confidence;
    the same window's expectation under the pitch grid becomes pitch_hz."""
    probs = F.softmax(logits, dim=-1)
    peak_bins = probs.argmax(dim=-1, keepdim=True)
    bin_indices = torch.arange(N_PITCH_BINS, device=probs.device).view(1, 1, -1)
    mask = (bin_indices >= peak_bins - PEAK_WIDTH) & (bin_indices <= peak_bins + PEAK_WIDTH)
    masked = probs * mask
    confidence = masked.sum(dim=-1)
    pitch_hz = (masked / (confidence.unsqueeze(-1) + 1e-8) * centers).sum(dim=-1)
    return pitch_hz, confidence


def load_weights(model: SwiftF0, weights_path: Path) -> None:
    state = load_file(str(weights_path))
    result = model.load_state_dict(state, strict=False)
    allowed_missing = {"window", "pitch_bin_centers"}
    missing = set(result.missing_keys)
    assert missing.issubset(allowed_missing), (
        f"unexpected missing keys: {missing - allowed_missing}"
    )
    assert not result.unexpected_keys, f"unexpected keys: {result.unexpected_keys}"


def capture_intermediates(model: SwiftF0, audio: torch.Tensor) -> dict[str, torch.Tensor]:
    """Forward pass with hooks on each conv block + freq_projection, returning
    every block's output alongside the input log-magnitude and final pitch/conf."""
    captured: dict[str, torch.Tensor] = {}
    handles = []
    layer_names = [f"conv{i}" for i in range(1, len(CHANNELS) + 1)] + ["freq_projection"]
    for name in layer_names:
        # bind name via default arg so each closure captures its own value.
        def hook(_module, _inputs, output, n=name):
            captured[n] = output
        handles.append(getattr(model, name).register_forward_hook(hook))
    try:
        with torch.no_grad():
            captured["log_mag"] = model.frontend(audio)
            pitch_hz, confidence = model(audio)
            captured["pitch_hz"] = pitch_hz
            captured["confidence"] = confidence
    finally:
        for h in handles:
            h.remove()
    return captured


def pick_slice(confidence: np.ndarray, n: int) -> int:
    """Return the start index of the n-frame window with highest mean confidence.
    Used to point the per-block fixture slice at a region with actual notes."""
    rolling = np.convolve(confidence, np.ones(n) / n, mode="valid")
    return int(np.argmax(rolling))


def dump_fixture(
    audio: np.ndarray,
    captured: dict[str, torch.Tensor],
) -> tuple[Path, int]:
    """Write tests/fixtures/swift-f0.safetensors. Whole-audio tensors cover the
    full clip end-to-end (audio, pitch_hz, confidence); per-block tensors are
    a SLICE_FRAMES window picked from the highest-confidence region so the
    slice exercises real notes."""
    confidence = captured["confidence"][0].numpy()
    pitch_hz = captured["pitch_hz"][0].numpy()
    start = pick_slice(confidence, SLICE_FRAMES)
    end = start + SLICE_FRAMES

    fixture: dict[str, np.ndarray] = {
        "audio_16k": np.ascontiguousarray(audio.astype(np.float32)),
        "pitch_hz": pitch_hz.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "slice_start": np.array(start, dtype=np.int64),
    }
    # log_mag, conv1..conv5, freq_projection: all (..., T) — slice the time axis.
    for name in ("log_mag", *(f"conv{i}" for i in range(1, len(CHANNELS) + 1)), "freq_projection"):
        tensor = captured[name].numpy()
        fixture[name] = np.ascontiguousarray(tensor[..., start:end])
    # Store freq_projection in logits layout (B, T, N) so it matches what our
    # forward() returns; the rest of the model captures stay in (B, C, ..., T).
    fixture["freq_projection"] = np.ascontiguousarray(fixture["freq_projection"].transpose(0, 2, 1))

    path = REPO_ROOT / "candle-swift-f0" / "tests" / "fixtures" / "swift-f0.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(fixture, str(path))
    return path, start


def report(label: str, ours: np.ndarray, theirs: np.ndarray) -> float:
    diff = float(np.abs(ours - theirs).max())
    print(f"  {label:30s} max abs diff {diff:.2e}")
    return diff


def verify(weights_path: Path) -> None:
    audio, _ = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE, mono=True)
    audio_t = torch.from_numpy(audio).unsqueeze(0)

    model = SwiftF0().eval()
    load_weights(model, weights_path)
    captured = capture_intermediates(model, audio_t)
    torch_pitch = captured["pitch_hz"][0].numpy()
    torch_conf = captured["confidence"][0].numpy()

    onnx_path = Path(swift_f0.__file__).parent / "model.onnx"
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {session.get_inputs()[0].name: audio[None, :].astype(np.float32)}
    ref_pitch, ref_conf = (a[0] for a in session.run(None, inputs))

    n_frames = len(torch_pitch)
    print(f"swift-f0: PyTorch parity on {AUDIO_PATH.name} ({n_frames} frames)\n")
    # Tolerances sit just above the fp32 noise floor of two independent STFT/conv
    # implementations: pre-log STFT magnitudes already agree to ~2e-5; log() then
    # amplifies near-zero bins, but subsequent convs and softmax damp the result
    # back down to ~2e-3 Hz on pitch and ~3e-6 on confidence.
    assert report("pitch_hz   vs onnx", torch_pitch, ref_pitch) < 2e-3
    assert report("confidence vs onnx", torch_conf, ref_conf) < 5e-6

    print("\nparity verified")

    fixture_path, slice_start = dump_fixture(audio, captured)
    rel = fixture_path.relative_to(REPO_ROOT)
    print(
        f"wrote fixture: {rel}\n"
        f"  full {n_frames} frames (audio_16k, pitch_hz, confidence)\n"
        f"  slice {SLICE_FRAMES} frames @ start={slice_start} "
        f"(log_mag, conv1..conv5, freq_projection)"
    )


def main() -> None:
    if len(sys.argv) != 2:
        prog = Path(sys.argv[0]).name
        print(f"usage: {prog} <weights.safetensors>", file=sys.stderr)
        sys.exit(2)
    verify(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
