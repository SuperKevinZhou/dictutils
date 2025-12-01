# Repository Guidelines

## Project Structure & Module Organization
- Library entrypoint in `src/lib.rs` with shared traits in `src/traits.rs`; format parsers live in `src/dict/` (mdict, stardict, zim, bgl, dsl).
- Indexing code is in `src/index/`; compression/encoding and test helpers live in `src/util/`.
- Tests sit in `src/tests/` (unit, integration, concurrent, performance, error, utils); examples in `examples/`; C++ reference implementations in `references/`. Keep build output in `target/` out of git.

## Build, Test, and Development Commands
- `cargo build` or `cargo build --all-features` compile the crate; `cargo doc --all-features --no-deps` inspects APIs.
- `cargo test` runs the suite; add `--features criterion performance_tests` for perf harnesses; `cargo bench --all-features` when Criterion is installed.
- `cargo fmt --check` and `cargo clippy --all-targets --all-features -D warnings` gate PRs; run `cargo audit` when available; `cargo run --example basic_dict_loading` for a quick demo.

## Coding Style & Naming Conventions
- Rust 2021 with rustfmt defaults (4-space indent, trailing commas); keep modules small and document invariants where they are not obvious.
- Module/file names stay snake_case; types/traits in UpperCamelCase; functions/variables snake_case; feature flags match `Cargo.toml` (`cli`, `rayon`, `high_performance`).
- Use the shared `Result<T>`/`DictError` from `traits.rs`; prefer `?` and `From` conversions instead of manual error mapping.

## Testing Guidelines
- Existing tests rely on mock dictionaries from `util::test_utils`; when changing parsers or index bounds add fixtures with real dictionaries (see `TASKS.md`).
- Add new cases under `src/tests/` with descriptive names (e.g., `should_validate_offsets`); extend concurrent/performance suites for threading or timing-sensitive work.
- Use `cargo test -- --nocapture` for debugging; property-style cases can use `proptest` to exercise malformed headers and compression windows.

## Commit & Pull Request Guidelines
- Commit subjects should be short, imperative, and optionally scoped (`dict: tighten offset checks`, `tests: add mdict fixture`); never commit `target/` or generated binaries.
- PRs need a problem/solution summary, linked issue or task, commands run, and notes on fixture coverage or benchmark deltas; add screenshots only when CLI output changes materially.

## Security & Format Handling
- Project remains experimental; parsers still lean on placeholder builders, so treat outputs as non-production and avoid trusting unvetted dictionaries.
- For format/compression changes, add offset/length validation, sanitize CLI/log output, and test against corrupted fixtures; run `cargo audit` before releases.
