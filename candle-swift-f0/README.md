# candle-swift-f0

SwiftF0 pitch detection in [candle](https://github.com/huggingface/candle).

A direct port of [SwiftF0](https://github.com/lars76/swift-f0) (Nieradzik, 2025), validated against the reference ONNX implementation for numerical parity.

## Weights

Pre-converted safetensors live at [`huggingface.co/jhartquist/swift-f0`](https://huggingface.co/jhartquist/swift-f0). Fetch on demand:

```rust
use candle_core::Device;
use candle_swift_f0::SwiftF0;

let model = SwiftF0::from_hub(&Device::Cpu)?;
```

Or load weights from a local path:

```rust
use candle_core::Device;
use candle_swift_f0::{SwiftF0, predict};

let bytes = std::fs::read("swift-f0.safetensors")?;
let model = SwiftF0::from_safetensors(&bytes, &Device::Cpu)?;
let frames = predict(&model, &audio)?;
```

## Features

- `hf-hub` (default): pull weights from the Hub via `from_hub(...)`. Disable to drop the network dependency.

## License

Dual-licensed under MIT or Apache-2.0.
