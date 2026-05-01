#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "basic-pitch[tf]",
#     "nnAudio>=0.3.4",
#     "numpy<2",
#     "safetensors>=0.4",
#     "setuptools<80",
#     "torch>=2.1",
# ]
# ///
"""
Export Basic Pitch TensorFlow weights, plus the nnAudio CQT2010v2 kernels,
to two separate safetensors files.

The split mirrors the Rust crate's runtime split between Frontend (CQT) and
BasicPitch (CNN), and matches the candle-crepe / candle-swift-f0 convention
of shipping only trained weights to the Hugging Face Hub:

  basic_pitch.safetensors  trained CNN weights (TF SavedModel → PyTorch layout)
      Conv2D kernels:  (kH, kW, in, out) -> (out, in, kH, kW)
      BatchNorm:       gamma/beta/moving_mean/moving_variance ->
                       weight/bias/running_mean/running_var
      bn1, bn2, bn3 + 6 conv2d layers; 16,864 parameters.

  cqt.safetensors          frozen CQT2010v2 buffers, fully derivable from
                           the model params (sr=22050, hop=256, fmin=27.5,
                           n_bins=309, bins_per_octave=36). Anyone with
                           nnAudio installed can regenerate them, so this
                           file is not uploaded to HF — it is the local
                           companion the Rust frontend loads at startup.
      cqt.kernel_real / cqt.kernel_imag  top-octave conv kernel bank
      cqt.lowpass                        inter-octave downsample FIR
      cqt.early_downsample_filter        optional one-shot upfront FIR
      cqt.lengths                        per-bin kernel length (sqrt scale)
      plus a few scalars (cqt.n_octaves, cqt.hop_length, cqt.n_fft,
      cqt.downsample_factor) that drive the per-octave loop.

Usage:
    ./export_safetensors.py <weights_dir>
    # writes <weights_dir>/basic_pitch.safetensors
    # and    <weights_dir>/cqt.safetensors
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file


SR = 22050
HOP = 256
FMIN = 27.5
N_BINS = 309
BPO = 36

# TF variable basename -> PyTorch attribute name.
_TF_BN_MAP = {
    "batch_normalization": "bn1",
    "batch_normalization_2": "bn2",
    "batch_normalization_3": "bn3",
}
_TF_CONV_MAP = {
    "conv2d_1": "contour_conv1",
    "contours-reduced": "contour_conv2",
    "conv2d_2": "note_conv1",
    "conv2d_3": "note_conv2",
    "conv2d_4": "onset_conv1",
    "conv2d_5": "onset_conv2",
}


def extract_cnn_weights() -> dict[str, np.ndarray]:
    """Pull TF SavedModel vars and convert to PyTorch layout."""
    import tensorflow as tf
    from basic_pitch import ICASSP_2022_MODEL_PATH

    saved = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
    tf_vars = {v.name: v.numpy() for v in saved.variables}

    out: dict[str, np.ndarray] = {}
    for tf_name, pt_name in _TF_BN_MAP.items():
        out[f"{pt_name}.weight"] = tf_vars[f"{tf_name}/gamma:0"]
        out[f"{pt_name}.bias"] = tf_vars[f"{tf_name}/beta:0"]
        out[f"{pt_name}.running_mean"] = tf_vars[f"{tf_name}/moving_mean:0"]
        out[f"{pt_name}.running_var"] = tf_vars[f"{tf_name}/moving_variance:0"]
    for tf_name, pt_name in _TF_CONV_MAP.items():
        kernel = tf_vars[f"{tf_name}/kernel:0"]  # (kH, kW, in, out)
        out[f"{pt_name}.weight"] = kernel.transpose(3, 2, 0, 1).copy()
        out[f"{pt_name}.bias"] = tf_vars[f"{tf_name}/bias:0"]

    n_params = sum(int(v.size) for v in out.values())
    assert n_params == 16_864, f"unexpected CNN param count: {n_params}"
    return out


def extract_cqt_kernels() -> dict[str, np.ndarray]:
    """Instantiate nnAudio CQT2010v2 with basic-pitch's exact params; pull
    its precomputed buffers and a couple of scalars the Rust frontend
    needs to drive the per-octave loop."""
    from nnAudio.features.cqt import CQT2010v2

    cqt = CQT2010v2(
        sr=SR,
        hop_length=HOP,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=BPO,
        pad_mode="reflect",
        verbose=False,
    )

    # top-octave kernels: shape (n_filters, 1, kernel_len). drop the conv1d
    # input-channel axis (always 1) for a clean 2D layout.
    kr = cqt.cqt_kernels_real.detach().cpu().numpy().squeeze(1)  # (n_filters, kernel_len)
    ki = cqt.cqt_kernels_imag.detach().cpu().numpy().squeeze(1)
    assert kr.ndim == 2 and ki.shape == kr.shape
    n_filters, kernel_len = kr.shape
    assert n_filters == BPO, f"unexpected top-octave filter count: {n_filters}"

    lowpass = cqt.lowpass_filter.detach().cpu().numpy().squeeze()
    assert lowpass.ndim == 1

    lengths = cqt.lenghts.detach().cpu().numpy().astype(np.float32)
    assert lengths.shape == (N_BINS,)

    out: dict[str, np.ndarray] = {
        "cqt.kernel_real": np.ascontiguousarray(kr.astype(np.float32)),
        "cqt.kernel_imag": np.ascontiguousarray(ki.astype(np.float32)),
        "cqt.lowpass": np.ascontiguousarray(lowpass.astype(np.float32)),
        "cqt.lengths": np.ascontiguousarray(lengths),
        # scalars (encoded as 1-element f32 tensors so they live in the
        # safetensors uniformly; Rust reads with .to_scalar::<f32>())
        "cqt.n_octaves": np.array([cqt.n_octaves], dtype=np.float32),
        "cqt.n_bins": np.array([N_BINS], dtype=np.float32),
        "cqt.n_fft": np.array([cqt.n_fft], dtype=np.float32),
        "cqt.hop_length": np.array([cqt.hop_length], dtype=np.float32),
        "cqt.downsample_factor": np.array([cqt.downsample_factor], dtype=np.float32),
    }

    if cqt.earlydownsample:
        eds = cqt.early_downsample_filter.detach().cpu().numpy().squeeze()
        assert eds.ndim == 1
        out["cqt.early_downsample_filter"] = np.ascontiguousarray(eds.astype(np.float32))

    return out


def write_safetensors(tensors: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    print(
        f"  wrote {len(tensors):>2} tensors to {path} "
        f"({path.stat().st_size / 1024:.1f} KB)"
    )


def export(weights_dir: Path) -> None:
    print("[1/2] CNN weights from TF SavedModel")
    cnn = extract_cnn_weights()
    n_params = sum(int(v.size) for v in cnn.values())
    print(f"  {len(cnn)} tensors, {n_params:,} trained params")
    write_safetensors(cnn, weights_dir / "basic_pitch.safetensors")

    print("\n[2/2] nnAudio CQT2010v2 kernels")
    cqt = extract_cqt_kernels()
    for k, v in cqt.items():
        print(f"  {k:35s} shape={tuple(v.shape)}")
    write_safetensors(cqt, weights_dir / "cqt.safetensors")


def main() -> None:
    if len(sys.argv) != 2:
        prog = Path(sys.argv[0]).name
        print(f"usage: {prog} <weights_dir>", file=sys.stderr)
        sys.exit(2)
    export(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
