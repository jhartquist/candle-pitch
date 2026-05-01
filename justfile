default:
    @just --list

# === Rust ===

build:
    cargo build --workspace

test:
    cargo test --release --workspace -- --test-threads=1

test-fast:
    cargo test --workspace --lib

fmt:
    cargo fmt --all

lint:
    cargo clippy --all-targets -- -D warnings

check: fmt lint test-fast

# === CREPE ===

# Export CREPE weights to safetensors. capacity in {tiny, small, medium, large, full}.
crepe-export capacity:
    ./candle-crepe/scripts/export_safetensors.py {{capacity}} candle-crepe/weights/{{capacity}}.safetensors

# Validate the safetensors via PyTorch reimpl against the TF reference, then
# dump the parity fixture to candle-crepe/tests/fixtures/{capacity}.safetensors.
crepe-pytorch-parity capacity:
    ./candle-crepe/scripts/pytorch_parity.py {{capacity}} candle-crepe/weights/{{capacity}}.safetensors

# Export weights and dump the parity fixture for one capacity.
crepe-prepare capacity: (crepe-export capacity) (crepe-pytorch-parity capacity)

# Export weights and dump parity fixtures for all five capacities.
crepe-prepare-all: \
    (crepe-prepare "tiny") \
    (crepe-prepare "small") \
    (crepe-prepare "medium") \
    (crepe-prepare "large") \
    (crepe-prepare "full")

# === SwiftF0 ===

# Export SwiftF0 weights to safetensors.
swift-f0-export:
    ./candle-swift-f0/scripts/export_safetensors.py candle-swift-f0/weights/swift-f0.safetensors

# Validate the safetensors via PyTorch reimpl against the ONNX reference, then
# dump the parity fixture to candle-swift-f0/tests/fixtures/swift-f0.safetensors.
swift-f0-pytorch-parity:
    ./candle-swift-f0/scripts/pytorch_parity.py candle-swift-f0/weights/swift-f0.safetensors

# Export weights and dump the parity fixture.
swift-f0-prepare: swift-f0-export swift-f0-pytorch-parity

# === CLI smoke tests ===

# Run the CLI with CREPE on data/AMajSlow.wav. Writes crepe.json + crepe.png.
run-crepe capacity="tiny":
    cargo run --release -p candle-pitch -- \
        crepe data/AMajSlow.wav --capacity {{capacity}} \
        --output crepe.json --plot crepe.png

# Run the CLI with Swift-F0 on data/AMajSlow.wav. Writes swift-f0.json + swift-f0.png.
run-swift-f0:
    cargo run --release -p candle-pitch -- \
        swift-f0 data/AMajSlow.wav \
        --output swift-f0.json --plot swift-f0.png
