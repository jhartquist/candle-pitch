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

# === Basic Pitch ===

# Export Basic Pitch TF weights and nnAudio CQT kernels to safetensors.
# Writes basic_pitch.safetensors (trained CNN, uploadable to HF) and
# cqt.safetensors (frozen CQT2010v2 buffers, regenerable from params).
basic-pitch-export:
    ./candle-basic-pitch/scripts/export_safetensors.py candle-basic-pitch/weights

# Validate the safetensors via PyTorch reimpl against the TF reference, then
# dump the parity fixture to candle-basic-pitch/tests/fixtures/basic_pitch.safetensors.
basic-pitch-pytorch-parity:
    ./candle-basic-pitch/scripts/pytorch_parity.py candle-basic-pitch/weights/basic_pitch.safetensors

# Export weights and dump the parity fixture.
basic-pitch-prepare: basic-pitch-export basic-pitch-pytorch-parity
