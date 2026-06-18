cd photo-editor-web
wasm-pack build --target web --release
cd ..
cargo about init
cargo about generate about.hbs > rust_licenses.html
