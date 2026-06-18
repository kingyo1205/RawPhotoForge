
# RawPhotoForge Rust+Godot

This directory contains the legacy Rust+Godot implementation of RawPhotoForge.

The contents are preserved for archival purposes and are no longer the primary development version.

## Rust+Godot Project Structure

```text
rust-godot-legacy/
├── photo-editor           # Image processing core
├── photo-editor-godot     # Godot GDExtension
└── raw-photo-forge        # Godot project (UI)
```

### photo-editor

* RAW image processing
* GPU image processing
* Metadata handling

### photo-editor-godot

* Integration layer between the Rust core and Godot

### raw-photo-forge

* User interface
* Image preview
* Settings

---

## Technology Stack

* Rust
* wgpu
* Godot Engine
* GDExtension

---

## About Godot Engine

* The user interface is built with **Godot Engine**
* Integration with Rust is implemented through **GDExtension**
* Godot Engine is distributed under the MIT License. See the official Godot website for details.

---

## Building

RawPhotoForge Desktop consists of:

* A core implemented in Rust
* A Godot-based UI using GDExtension

The following instructions describe how to build the project from source on Linux.

### Requirements

* Rust
* Cargo
* Godot Engine (standard version, .NET version not required)
* Linux (tested on x86_64)

### Build Steps

```bash
git clone https://github.com/kingyo1205/RawPhotoForge.git
cd RawPhotoForge/rust-godot-legacy

cargo build -r -p photo-editor-godot
cargo about init
cargo about generate about.hbs > rust_licenses.html
mkdir ./raw-photo-forge/addons/photo_editor/libs
mkdir ./raw-photo-forge/addons/photo_editor/libs/Linux-x86_64
cp ./target/release/libphoto_editor_godot.so ./raw-photo-forge/addons/photo_editor/libs/Linux-x86_64/libphoto_editor_godot.so
```

### Exporting with Godot

1. Open `rust-godot-legacy/raw-photo-forge/` as a project in Godot.
2. Verify that the GDExtension is loaded correctly.
3. Use Godot's export feature to generate application binaries.

---

## License

### Project License

This project is licensed under the

**GNU Affero General Public License v3.0 (AGPL-3.0)**

See the `LICENSE` file for details.

### Third-Party Dependency Licenses (Rust)

A list of licenses for Rust dependencies is available in:

* `rust-godot-legacy/rust_licenses.html`

This file is generated using `cargo-about`.

Example:

```bash
cargo about generate about.hbs > rust_licenses.html
```

---

## Tested Environment

* Linux x86_64

Other platforms have not been tested.


