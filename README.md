# Wasm-Parser

To build for release:

```
wasm-pack build --scope novorender --release
```

To build with a watch that rebuilds on save:

```
cargo watch -i .gitignore -i "pkg/*" -s "wasm-pack build --scope novorender --release --features=console"
```

At least on linux that line usually makes the npm watch to fail, The following alternative generates the files on a different folder and then copes them to the write location which fixes the problem, a similar command could be used in windows to finally move the files and delete the temporary folder:

```
cargo watch -i .gitignore -i "pkg/*" -i "pkg_new" -s "wasm-pack build --scope novorender --release --features=console --out-dir pkg_new && mv pkg_new/* pkg/ && rm -rf pkg_new"
```

To build with debug symbols that can be inspected with chromes WebAssembly DWARF extension;

```
cargo build --features=console --profile=release --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/debug-optimized/wasm_parser.wasm --debug --keep-debug --out-dir pkg
```

## Available features:

- console: enables log! macro to log to the browser console
- checked_types: enables enum checks while parsing for debugging that the parser is working and
files are correct
- memory-monitor: enables logging of used memory to the browser console on every call to geometry
and every drop of a schema. Useful to detect memory leaks