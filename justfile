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
