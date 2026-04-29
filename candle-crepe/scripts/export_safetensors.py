#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["crepe", "h5py", "numpy", "safetensors"]
# ///
"""
Export CREPE TensorFlow weights to safetensors.

Locates the .h5 from the installed crepe package (uv handles install on
first run), reshapes conv kernels and dense weights to the standard PyTorch
layout, and writes a flat safetensors file readable by any framework.

Tensor names:
    conv{i}.conv.{weight,bias}                          i in 1..=6
    conv{i}.bn.{weight,bias,running_mean,running_var}   i in 1..=6
    classifier.{weight,bias}

Usage:
    ./export.py <capacity> <out.safetensors>

Where <capacity> is one of: tiny, small, medium, large, full.
"""

import sys
from pathlib import Path

import crepe
import h5py
import numpy as np
from safetensors.numpy import save_file


CAPACITIES = ("tiny", "small", "medium", "large", "full")

# six conv + BN blocks and a final dense classifier.
EXPECTED_LAYERS = {
    *(f"conv{i}" for i in range(1, 7)),
    *(f"conv{i}-BN" for i in range(1, 7)),
    "classifier",
}

BN_PARAMS = {
    "gamma:0": "weight",
    "beta:0": "bias",
    "moving_mean:0": "running_mean",
    "moving_variance:0": "running_var",
}


def remap(layer: str, param: str, tensor: np.ndarray) -> tuple[str, np.ndarray]:
    if layer.endswith("-BN"):
        block = layer.removesuffix("-BN")
        return f"{block}.bn.{BN_PARAMS[param]}", tensor

    if param == "kernel:0":
        if tensor.ndim == 4:
            # Conv2D (kw, kh, in, out) with kh=1 -> Conv1d (out, in, kw)
            kw, kh, ci, co = tensor.shape
            assert kh == 1, f"expected kh=1, got {kh}"
            return f"{layer}.conv.weight", tensor.reshape(kw, ci, co).transpose(2, 1, 0)
        # Dense (in, out) -> Linear (out, in)
        return f"{layer}.weight", tensor.T

    if param == "bias:0":
        if layer == "classifier":
            return f"{layer}.bias", tensor
        return f"{layer}.conv.bias", tensor

    raise ValueError(f"unexpected tensor: {layer}/{param}")


def export(capacity: str, out_path: Path) -> None:
    h5_path = Path(crepe.__file__).parent / f"model-{capacity}.h5"
    weights = {}

    with h5py.File(h5_path, "r") as f:
        # keras nests datasets at irregular depth (variable names contain
        # slashes that h5py turns into groups), so walk the whole file and
        # take the first/last path components as layer/param.
        datasets = []

        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                parts = name.split("/")
                datasets.append((parts[0], parts[-1], np.asarray(obj, dtype=np.float32)))

        f.visititems(collect)

        weighted = {layer for layer, _, _ in datasets}
        assert weighted == EXPECTED_LAYERS, (
            f"weighted layer mismatch in {h5_path.name}: "
            f"missing={sorted(EXPECTED_LAYERS - weighted)}, "
            f"unexpected={sorted(weighted - EXPECTED_LAYERS)}"
        )

        for layer, param, tensor in datasets:
            name, tensor = remap(layer, param, tensor)
            weights[name] = np.ascontiguousarray(tensor)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(out_path))
    print(f"wrote {len(weights)} tensors to {out_path}")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in CAPACITIES:
        prog = Path(sys.argv[0]).name
        caps = "|".join(CAPACITIES)
        print(f"usage: {prog} <{caps}> <out.safetensors>", file=sys.stderr)
        sys.exit(2)
    export(sys.argv[1], Path(sys.argv[2]))


if __name__ == "__main__":
    main()
