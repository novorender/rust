# RUSTFLAGS='-C target-feature=+simd128,+bulk-memory,+mutable-globals -C link-arg=--import-memory' cargo +stable build --profile=debug-optimized --target=wasm32-unknown-unknown --features=console
RUSTFLAGS='-C target-feature=+simd128,+bulk-memory,+mutable-globals -C link-arg=--import-memory' cargo +stable build --profile=release --target=wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/release/wasm_parser.wasm --target=web --out-dir pkg_new #--keep-debug --debug
../wasm-shared-mem/target/debug/wasm-shared-mem --in-dir pkg_new --out-dir pkg_new --pkg-name wasm_parser