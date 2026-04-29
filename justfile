default:
    @just --list

# === Rust ===

build:
    cargo build --workspace

test:
    cargo test --workspace

fmt:
    cargo fmt --all

lint:
    cargo clippy --all-targets -- -D warnings

check: fmt lint test

# === CREPE ===

# Export CREPE weights to safetensors. capacity in {tiny, small, medium, large, full}.
crepe-export capacity:
    ./candle-crepe/scripts/export_safetensors.py {{capacity}} candle-crepe/weights/{{capacity}}.safetensors

# Validate the safetensors via PyTorch reimpl against the TF reference, then
# dump the parity fixture to candle-crepe/tests/fixtures/{capacity}.safetensors.
crepe-pytorch-parity capacity:
    ./candle-crepe/scripts/pytorch_parity.py {{capacity}} candle-crepe/weights/{{capacity}}.safetensors
