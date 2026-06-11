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

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner is the standalone apidoc/ package, so it is never
# built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Local CI sanity check
ci: fmt clippy feature-check test test-nostd

# Run benchmarks
bench *ARGS:
    cargo bench {{ARGS}}
