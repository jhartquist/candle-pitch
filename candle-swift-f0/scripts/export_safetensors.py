#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["swift-f0", "onnx", "numpy", "safetensors"]
# ///
"""
Export SwiftF0 ONNX weights to safetensors.

Locates the .onnx from the installed swift-f0 package (uv handles install
on first run), walks the graph in topological order, and renames the Conv
initializers under canonical names. The bundled ONNX is already inference-
fused (Conv-ReLU, no BN), so the export is a pure rename pass.

Tensor names:
    conv{i}.{weight,bias}           i in 1..=5
    freq_projection.{weight,bias}

Usage:
    ./export_safetensors.py <out.safetensors>
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import swift_f0
from safetensors.numpy import save_file


# each entry is (output_name, expected_weight_shape). the list is in graph
# topological order so it can be zipped against the Conv nodes directly.
EXPECTED_LAYERS = [
    ("conv1", (8, 1, 5, 5)),
    ("conv2", (16, 8, 5, 5)),
    ("conv3", (32, 16, 5, 5)),
    ("conv4", (64, 32, 5, 5)),
    ("conv5", (1, 64, 5, 5)),
    ("freq_projection", (200, 132, 1)),
]


def export(out_path: Path) -> None:
    onnx_path = Path(swift_f0.__file__).parent / "model.onnx"
    model = onnx.load(str(onnx_path))
    initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

    # model.graph.node is topologically ordered; Conv nodes line up 1:1 with EXPECTED_LAYERS.
    conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]
    assert len(conv_nodes) == len(EXPECTED_LAYERS), (
        f"expected {len(EXPECTED_LAYERS)} Conv nodes, got {len(conv_nodes)}"
    )

    weights: dict[str, np.ndarray] = {}
    for node, (name, expected_shape) in zip(conv_nodes, EXPECTED_LAYERS):
        assert len(node.input) == 3, f"{name}: Conv has no bias input"
        w = initializers[node.input[1]]
        b = initializers[node.input[2]]
        assert w.shape == expected_shape, f"{name}.weight shape {w.shape} != {expected_shape}"
        weights[f"{name}.weight"] = np.ascontiguousarray(w)
        weights[f"{name}.bias"] = np.ascontiguousarray(b)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(out_path))
    print(f"wrote {len(weights)} tensors to {out_path}")


def main() -> None:
    if len(sys.argv) != 2:
        prog = Path(sys.argv[0]).name
        print(f"usage: {prog} <out.safetensors>", file=sys.stderr)
        sys.exit(2)
    export(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
