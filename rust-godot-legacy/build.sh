cargo build -r -p photo-editor-godot
cargo about init
cargo about generate about.hbs > rust_licenses.html
mkdir ./raw-photo-forge/addons/photo_editor/libs
mkdir ./raw-photo-forge/addons/photo_editor/libs/Linux-x86_64
cp ./target/release/libphoto_editor_godot.so ./raw-photo-forge/addons/photo_editor/libs/Linux-x86_64/libphoto_editor_godot.so