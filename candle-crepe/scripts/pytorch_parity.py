#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["crepe", "librosa", "numpy", "safetensors", "tensorflow", "torch"]
# ///
"""
PyTorch reimplementation of CREPE, plus parity check and fixture dump.

Defines the model as Conv1d + BatchNorm1d (matching the safetensors layout
produced by export_safetensors.py), loads weights, and verifies bit parity
against the TF reference on the project's test audio: activations, plus
decoded frequency and confidence both with and without viterbi.

Once parity holds, dumps a small safetensors fixture (resampled audio,
normalized frames, per-block activations, final salience, decoded outputs)
under tests/fixtures/<capacity>.safetensors. The Rust parity tests consume
this file and compare layer by layer.

Usage:
    ./pytorch_parity.py <capacity> <weights.safetensors>
"""

import sys
from pathlib import Path

import crepe
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file
from safetensors.torch import load_file


CAPACITIES = ("tiny", "small", "medium", "large", "full")
MULTIPLIER = {"tiny": 4, "small": 8, "medium": 16, "large": 24, "full": 32}
FILTERS = (32, 4, 4, 4, 8, 16)
KERNELS = (512, 64, 64, 64, 64, 64)
STRIDES = (4, 1, 1, 1, 1, 1)
PADDINGS = ((254, 254), (31, 32), (31, 32), (31, 32), (31, 32), (31, 32))
PITCH_BINS = 360
SAMPLE_RATE = 16000
FRAME_LEN = 1024
HOP = 160

# size of the slice used for per-block intermediate fixtures. Per-block tensors
# scale with frame count, so we cap them at a small slice; the slice is picked
# in a high-confidence region so it covers actual notes, not silence/attack.
SLICE_FRAMES = 16

# bin index -> cents. bin 0 corresponds to ~32.7 Hz (C1) and bins are spaced
# 20 cents apart up to bin 359.
CENTS_MAPPING = np.linspace(0, 7180, PITCH_BINS) + 1997.3794084376191

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIO_PATH = REPO_ROOT / "data" / "AMajSlow.wav"


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel, stride)
        # eps=1e-3 matches TF/Keras BatchNormalization default
        self.bn = torch.nn.BatchNorm1d(out_channels, eps=1e-3)

    def forward(self, x: torch.Tensor, padding: tuple[int, int]) -> torch.Tensor:
        x = F.pad(x, padding)
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)
        return F.max_pool1d(x, kernel_size=2, stride=2)


class Crepe(torch.nn.Module):
    def __init__(self, capacity: str):
        super().__init__()
        mult = MULTIPLIER[capacity]
        out_channels = [f * mult for f in FILTERS]
        in_channels = [1] + out_channels[:-1]
        for i in range(6):
            self.add_module(
                f"conv{i + 1}",
                ConvBlock(in_channels[i], out_channels[i], KERNELS[i], STRIDES[i]),
            )
        # after stride-4 conv1 + 6 maxpool(2): time axis collapses 1024 -> 4
        self.classifier = torch.nn.Linear(4 * out_channels[-1], PITCH_BINS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(6):
            x = getattr(self, f"conv{i + 1}")(x, PADDINGS[i])
        # TF flattens time-major after Permute((2,1,3)); PyTorch needs (B, T, C) first.
        x = x.permute(0, 2, 1).flatten(1)
        return torch.sigmoid(self.classifier(x))


def load_weights(model: Crepe, weights_path: Path) -> None:
    state = load_file(str(weights_path))
    result = model.load_state_dict(state, strict=False)
    allowed_missing = {f"conv{i}.bn.num_batches_tracked" for i in range(1, 7)}
    missing = set(result.missing_keys)
    assert missing.issubset(allowed_missing), (
        f"unexpected missing keys: {missing - allowed_missing}"
    )
    assert not result.unexpected_keys, f"unexpected keys: {result.unexpected_keys}"


def make_frames(audio: np.ndarray) -> np.ndarray:
    """Pad, frame at 10ms hop, and per-frame normalize. Mirrors crepe.core.get_activation."""
    audio = np.pad(audio, FRAME_LEN // 2, mode="constant")
    n = 1 + (len(audio) - FRAME_LEN) // HOP
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n, FRAME_LEN),
        strides=(HOP * audio.itemsize, audio.itemsize),
    ).copy()
    frames -= frames.mean(axis=1, keepdims=True)
    frames /= np.clip(frames.std(axis=1, keepdims=True), 1e-8, None)
    return frames


def decode(activation: np.ndarray, viterbi: bool) -> tuple[np.ndarray, np.ndarray]:
    """activation -> (frequency Hz, confidence). Pure numpy."""
    cents = (
        to_viterbi_cents(activation) if viterbi else to_local_average_cents(activation)
    )
    frequency = 10.0 * 2.0 ** (cents / 1200.0)
    frequency[np.isnan(frequency)] = 0
    confidence = activation.max(axis=1)
    return frequency, confidence


def to_local_average_cents(
    salience: np.ndarray, centers: np.ndarray | None = None
) -> np.ndarray:
    """Weighted-average cents in a +/-4 bin window around argmax (or given) bin."""
    if centers is None:
        centers = np.argmax(salience, axis=1)
    cents = np.empty(len(salience))
    for t in range(len(salience)):
        center = int(centers[t])
        lo = max(0, center - 4)
        hi = min(PITCH_BINS, center + 5)
        weights = salience[t, lo:hi]
        cents[t] = (weights * CENTS_MAPPING[lo:hi]).sum() / weights.sum()
    return cents


def to_viterbi_cents(salience: np.ndarray) -> np.ndarray:
    """HMM-smoothed bin path via Viterbi, then cents via local-average around the path."""
    start, transition, emission = _hmm_params()
    observations = np.argmax(salience, axis=1).astype(np.int64)
    path = _viterbi(observations, start, transition, emission)
    return to_local_average_cents(salience, centers=path)


def _hmm_params() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = np.full(PITCH_BINS, 1.0 / PITCH_BINS)
    # transition: tent function (12 - |i - j|, clamped at 0), normalized per row.
    bins = np.arange(PITCH_BINS)
    bin_distance = np.abs(bins[:, None] - bins[None, :])
    transition = np.maximum(12.0 - bin_distance, 0.0)
    transition /= transition.sum(axis=1, keepdims=True)
    # emission: each state observes its own bin with extra weight 0.1, uniform otherwise.
    p_self = 0.1
    emission = np.full((PITCH_BINS, PITCH_BINS), (1.0 - p_self) / PITCH_BINS)
    np.fill_diagonal(emission, p_self + (1.0 - p_self) / PITCH_BINS)
    return start, transition, emission


def _viterbi(
    observations: np.ndarray,
    start: np.ndarray,
    transition: np.ndarray,
    emission: np.ndarray,
) -> np.ndarray:
    """Standard Viterbi decoding in log space.

    Mirrors hmmlearn's _hmmc.viterbi: forward stores only the lattice;
    backward reconstructs the path by re-running the max. Tie-breaking matches
    hmmlearn's quirk: first-index for the final frame (via std::max_element),
    last-index for all earlier frames (via std::max on (value, index) pairs).
    """
    n_frames, n_states = len(observations), len(start)
    with np.errstate(divide="ignore"):
        log_start = np.log(start)
        log_trans = np.log(transition)
        log_emit = np.log(emission)

    log_prob = np.full((n_frames, n_states), -np.inf)
    log_prob[0] = log_start + log_emit[:, observations[0]]
    for t in range(1, n_frames):
        scores = log_prob[t - 1, :, None] + log_trans
        log_prob[t] = np.max(scores, axis=0) + log_emit[:, observations[t]]

    path = np.empty(n_frames, dtype=np.int64)
    path[-1] = int(np.argmax(log_prob[-1]))
    for t in range(n_frames - 2, -1, -1):
        scores = log_prob[t] + log_trans[:, path[t + 1]]
        # Last-index argmax to match hmmlearn's pair-based tie-breaking.
        path[t] = (n_states - 1) - int(np.argmax(scores[::-1]))
    return path


def capture_intermediates(model: Crepe, frames: torch.Tensor) -> dict[str, torch.Tensor]:
    """Forward pass with a hook on each conv block, returning every block's
    post-pool output alongside the input frames and the final salience."""
    captured: dict[str, torch.Tensor] = {"frames": frames}
    handles = []
    for i in range(1, 7):
        block = getattr(model, f"conv{i}")
        # bind name via default arg so each closure captures its own value.
        def hook(_module, _inputs, output, name=f"conv{i}"):
            captured[name] = output
        handles.append(block.register_forward_hook(hook))
    try:
        with torch.no_grad():
            captured["salience"] = model(frames)
    finally:
        for h in handles:
            h.remove()
    return captured


def pick_slice(salience: np.ndarray, n: int) -> int:
    """Return the start index of the n-frame window with highest mean confidence.
    Used to point the per-block fixture slice at a region with actual notes."""
    confidence = salience.max(axis=1)
    rolling = np.convolve(confidence, np.ones(n) / n, mode="valid")
    return int(np.argmax(rolling))


def dump_fixture(
    capacity: str,
    audio: np.ndarray,
    model: Crepe,
    frames: np.ndarray,
    salience: np.ndarray,
) -> tuple[Path, int]:
    """Write tests/fixtures/<capacity>.safetensors. Whole-audio tensors cover
    the full clip end-to-end (resample, salience, decoded freq/conf); per-block
    tensors cover a SLICE_FRAMES window picked from the highest-confidence
    region so the slice exercises real notes."""
    full_freq_local, full_conf = decode(salience, viterbi=False)
    full_freq_viterbi, _ = decode(salience, viterbi=True)

    start = pick_slice(salience, SLICE_FRAMES)
    end = start + SLICE_FRAMES
    captured = capture_intermediates(
        model, torch.from_numpy(frames[start:end]).unsqueeze(1)
    )
    # full salience supersedes the slice's; drop the duplicate.
    captured.pop("salience")

    fixture: dict[str, np.ndarray] = {
        k: np.ascontiguousarray(v.numpy()) for k, v in captured.items()
    }
    fixture["audio_16k"] = np.ascontiguousarray(audio.astype(np.float32))
    fixture["salience"] = salience.astype(np.float32)
    fixture["frequency_local"] = full_freq_local.astype(np.float32)
    fixture["frequency_viterbi"] = full_freq_viterbi.astype(np.float32)
    fixture["confidence"] = full_conf.astype(np.float32)

    path = REPO_ROOT / "candle-crepe" / "tests" / "fixtures" / f"{capacity}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(fixture, str(path))
    return path, start


def report(label: str, ours: np.ndarray, theirs: np.ndarray) -> float:
    diff = float(np.abs(ours - theirs).max())
    print(f"  {label:30s} max abs diff {diff:.2e}")
    return diff


def verify(capacity: str, weights_path: Path) -> None:
    audio, _ = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE, mono=True)
    frames = make_frames(audio)

    model = Crepe(capacity).eval()
    load_weights(model, weights_path)
    with torch.no_grad():
        torch_act = model(torch.from_numpy(frames).unsqueeze(1)).numpy()

    _, ref_freq_local, ref_conf, tf_act = crepe.predict(
        audio, SAMPLE_RATE, capacity, viterbi=False, verbose=0,
    )
    _, ref_freq_viterbi, _, _ = crepe.predict(
        audio, SAMPLE_RATE, capacity, viterbi=True, verbose=0,
    )
    our_freq_local, our_conf = decode(tf_act, viterbi=False)
    our_freq_viterbi, _ = decode(tf_act, viterbi=True)

    print(f"crepe-{capacity}: PyTorch parity on {AUDIO_PATH.name} ({len(frames)} frames)\n")

    # model parity: torch produces TF-equivalent activations.
    assert report("activations vs tf", torch_act, tf_act) < 1e-4
    # decode parity: our cents-and-frequency match crepe given the same activations.
    assert report("local freq  vs crepe", our_freq_local, ref_freq_local) < 1e-2
    assert report("local conf  vs crepe", our_conf, ref_conf) < 1e-4

    # Cents diff against hmmlearn (bit-exact when our viterbi matches its
    # asymmetric tie-breaking: first-index for the final frame, last-index for
    # earlier frames). Bins are 20c apart, so any non-zero diff is a path flip.
    diff_cents = 1200.0 * np.abs(np.log2(our_freq_viterbi / ref_freq_viterbi))
    worst = diff_cents.max()
    print(f"  {'viterbi vs hmmlearn':30s} max abs diff {worst:.2e}c")
    assert worst < 1e-6, f"viterbi diverges from hmmlearn ({worst:.2f}c)"

    print("\nparity verified")

    fixture_path, slice_start = dump_fixture(capacity, audio, model, frames, torch_act)
    rel = fixture_path.relative_to(REPO_ROOT)
    print(
        f"wrote fixture: {rel}\n"
        f"  full {len(frames)} frames (audio_16k, salience, freq_*, confidence)\n"
        f"  slice {SLICE_FRAMES} frames @ start={slice_start} (frames, conv1..conv6)"
    )


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in CAPACITIES:
        prog = Path(sys.argv[0]).name
        caps = "|".join(CAPACITIES)
        print(f"usage: {prog} <{caps}> <weights.safetensors>", file=sys.stderr)
        sys.exit(2)
    verify(sys.argv[1], Path(sys.argv[2]))


if __name__ == "__main__":
    main()
