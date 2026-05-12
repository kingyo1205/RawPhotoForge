# RawPhotoForge

GPU Photo Editor Project

![Screenshot](images/screenshot-main.png)

RawPhotoForge is a photo editing project
that utilizes GPU acceleration.

Currently:

* WebGPU-based Web version
* Rust + Godot-based Desktop version

are under development.

The project focuses on
high performance,
non-destructive editing,
and cross-platform design.

---

## Repository Structure

```text
RawPhotoForge/
├── web/              # Web version
├── rust/             # Rust + Godot Desktop version
└── python-legacy/    # Legacy Python version
```

### web

* WebGPU-based
* TypeScript
* Runs in the browser
* Static hosting

### rust

* Core: Rust
* GPU: wgpu
* UI: Godot Engine + GDExtension
* RAW support

### python-legacy

* Initial implementation
* Python-based
* Preserved for existing releases
* No longer actively maintained

---

## Projects

### RawPhotoForge Web

A browser-based GPU photo editing application.

#### Features

* WebGPU-based
* TypeScript
* Static hosting
* No installation required
* Cross-platform

#### Tech Stack

* TypeScript
* WebGPU

Deno is used for development.

---

### RawPhotoForge Desktop

A native RAW development application
built with Rust + Godot.

#### Features

* RAW support
* Rust
* wgpu
* Godot Engine
* GDExtension
* GPU acceleration
* Non-destructive editing

---

#### Rust Version Structure

```text
rust/
├── photo-editor           # Image processing core
├── photo-editor-godot     # GDExtension for Godot
└── raw-photo-forge        # Godot project (UI)
```

##### photo-editor

* RAW image processing
* GPU image operations
* Metadata processing

##### photo-editor-godot

* Integration layer between the Rust core and Godot

##### raw-photo-forge

* UI
* Preview
* Various settings

---

#### Tech Stack

* Rust
* wgpu
* Godot Engine
* GDExtension

---

#### About Godot Engine

* The UI uses **Godot Engine**
* Integrated with Rust via **GDExtension**
* Godot Engine itself is distributed under the MIT License. See the official Godot website for details

---

#### Build Instructions

RawPhotoForge Desktop consists of:
a Rust-based core and
a UI built with Godot(GDExtension).

Below are the steps for building from source
on a Linux environment.

##### Requirements

* Rust
* Cargo
* Godot Engine (standard version, .NET version not required)
* Linux (tested on x86_64)

##### Steps

```bash
git clone https://github.com/kingyo1205/RawPhotoForge.git
cd RawPhotoForge

# Build the Rust GDExtension
cargo build -r -p photo-editor-godot

# Generate dependency license information
cargo about init
cargo about generate about.hbs > rust_licenses.html

# Copy the generated shared library into the Godot addon
cp ./target/release/libphoto_editor_godot.so ./rust/raw-photo-forge/addons/photo_editor/libs/Linux-x86_64/
```

##### Exporting with Godot

1. Open the `rust/raw-photo-forge/` directory as a project in Godot
2. Confirm that the GDExtension is loaded correctly
3. Use Godot's export feature to generate binaries

---

## License

### Main Project

This project is released under the
**GNU Affero General Public License v3.0 (AGPL-3.0)**.

See the `LICENSE` file for details.

### Dependency Licenses (Rust)

A list of licenses for the dependencies
used in the Rust version is provided in:

* `rust_licenses.html`

This file is generated using
`cargo-about`.

Example:

```bash
cargo about generate about.hbs > rust_licenses.html
```

---

## AI Tools Used

The following AI tools are used
during the development of RawPhotoForge.

* ChatGPT
* Gemini
* Gemini CLI
* Claude

---

## Philosophy

* GPU accelerated
* GPU vendor-free
* Cross-platform
* Non-destructive editing
* Local-first
* Open source

---

## Notes

* The project is being developed with the goal of becoming an alternative to commercial RAW editing software
* The architecture and implementation are continuously evolving

---

## Development Status

* Python version: Development ended
* Web version: Under active development
* Rust version: Under active development
