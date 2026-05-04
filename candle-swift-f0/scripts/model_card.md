---
license: mit
library_name: candle
tags:
  - audio
  - pitch-detection
  - pitch-estimation
  - candle
  - safetensors
---

# SwiftF0 (safetensors)

Compact CNN pitch tracker from Nieradzik, 2025 (arXiv:2508.18440). Original implementation: [lars76/swift-f0](https://github.com/lars76/swift-f0).

This repository hosts the released SwiftF0 weights converted from the upstream ONNX model to `safetensors` for use with the [`candle`](https://github.com/huggingface/candle) ML framework via [`candle-swift-f0`](https://github.com/jhartquist/candle-pitch/tree/main/candle-swift-f0).

## Files

| File | Approx. size |
| --- | --- |
| `swift-f0.safetensors` | 375 KB |

The bundled ONNX is already inference-fused (Conv plus ReLU, no BatchNorm), so the conversion is a pure rename pass that preserves the original tensor shapes.

Names:

```
conv{i}.{weight,bias}           i in 1..=5
freq_projection.{weight,bias}
```

## Provenance

Converted from the bundled `model.onnx` of the [`swift-f0`](https://pypi.org/project/swift-f0/) PyPI package using [`scripts/export_safetensors.py`](https://github.com/jhartquist/candle-pitch/blob/main/candle-swift-f0/scripts/export_safetensors.py).

## Parity

The candle implementation reproduces the reference ONNX forward pass to within `1e-4` max absolute difference on logits and on decoded pitch. Verification runs in [`scripts/pytorch_parity.py`](https://github.com/jhartquist/candle-pitch/blob/main/candle-swift-f0/scripts/pytorch_parity.py) and in the Rust integration tests under `candle-swift-f0/tests/`.

## Citation

```bibtex
@misc{nieradzik2025swiftf0,
  title={SwiftF0: Fast and Accurate Monophonic Pitch Detection},
  author={Lars Nieradzik},
  year={2025},
  eprint={2508.18440},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2508.18440}
}
```

## License

Same as upstream SwiftF0: MIT, Copyright (c) 2025 Lars Nieradzik. See `LICENSE`.
