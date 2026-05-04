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

# CREPE (safetensors)

Convolutional pitch tracker from Kim et al., ICASSP 2018. Original implementation: [marl/crepe](https://github.com/marl/crepe).

This repository hosts the five published capacities (`tiny`, `small`, `medium`, `large`, `full`), converted from the upstream TensorFlow weights to `safetensors` for use with the [`candle`](https://github.com/huggingface/candle) ML framework via [`candle-crepe`](https://github.com/jhartquist/candle-pitch/tree/main/candle-crepe).

## Files

| File | Capacity multiplier | Approx. size |
| --- | --- | --- |
| `tiny.safetensors` | 4 | 1.9 MB |
| `small.safetensors` | 8 | 6.2 MB |
| `medium.safetensors` | 16 | 23 MB |
| `large.safetensors` | 24 | 49 MB |
| `full.safetensors` | 32 | 85 MB |

Tensor layout follows PyTorch conventions. Convolutions are stored as `Conv1d (out, in, kernel)`, the dense classifier as `Linear (out, in)`, and BatchNorm parameters are split into `weight`, `bias`, `running_mean`, `running_var`.

Names:

```
conv{i}.conv.{weight,bias}                          i in 1..=6
conv{i}.bn.{weight,bias,running_mean,running_var}   i in 1..=6
classifier.{weight,bias}
```

## Provenance

Converted from the bundled `.h5` weights of the [`crepe`](https://pypi.org/project/crepe/) PyPI package using [`scripts/export_safetensors.py`](https://github.com/jhartquist/candle-pitch/blob/main/candle-crepe/scripts/export_safetensors.py).

## Parity

Each capacity reproduces the reference TensorFlow forward pass to within `1e-4` max absolute difference on the per-bin activation matrix and on decoded pitch. Verification runs in [`scripts/pytorch_parity.py`](https://github.com/jhartquist/candle-pitch/blob/main/candle-crepe/scripts/pytorch_parity.py) and in the Rust integration tests under `candle-crepe/tests/`.

## Citation

```bibtex
@inproceedings{kim2018crepe,
  title={CREPE: A Convolutional Representation for Pitch Estimation},
  author={Kim, Jong Wook and Salamon, Justin and Li, Peter and Bello, Juan Pablo},
  booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={161--165},
  year={2018},
  organization={IEEE}
}
```

## License

Same as upstream CREPE: MIT, Copyright (c) 2018 Jong Wook Kim. See `LICENSE`.
