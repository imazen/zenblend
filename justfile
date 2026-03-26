# zenblend development tasks

# Run tests (default features)
test:
    cargo test

# Run tests (no_std)
test-nostd:
    cargo test --no-default-features

# Check all feature permutations
feature-check:
    cargo check
    cargo check --no-default-features

# Clippy
clippy:
    cargo clippy --all-targets -- -D warnings

# Format
fmt:
    cargo fmt

# Local CI sanity check
ci: fmt clippy feature-check test test-nostd

# Run benchmarks
bench *ARGS:
    cargo bench {{ARGS}}
