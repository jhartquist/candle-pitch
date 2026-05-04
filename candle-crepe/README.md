# candle-crepe

CREPE pitch detection in [candle](https://github.com/huggingface/candle).

A direct port of the [original MARL/CREPE](https://github.com/marl/crepe) (Kim et al., ICASSP 2018), validated against the reference TensorFlow implementation for numerical parity.

## Weights

Pre-converted safetensors live at [`huggingface.co/jhartquist/crepe`](https://huggingface.co/jhartquist/crepe), one file per capacity (`tiny`, `small`, `medium`, `large`, `full`). Fetch on demand:

```rust
use candle_core::Device;
use candle_crepe::{Capacity, Crepe};

let model = Crepe::from_hub(Capacity::Tiny, &Device::Cpu)?;
```

Or load weights from a local path:

```rust
use candle_core::Device;
use candle_crepe::{Crepe, Decoder, predict};

let bytes = std::fs::read("tiny.safetensors")?;
let model = Crepe::from_safetensors(&bytes, &Device::Cpu)?;
let frames = predict(&model, &audio, Decoder::Viterbi)?;
```

## Features

- `hf-hub` (default): pull weights from the Hub via `from_hub(...)`. Disable to drop the network dependency.

## License

Dual-licensed under MIT or Apache-2.0.
