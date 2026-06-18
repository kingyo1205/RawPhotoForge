# RawPhotoForge

GPU Photo Editor Project

### [⭐Try RawPhotoForge⭐](https://kingyo1205.github.io/RawPhotoForge/web/dist/index.html)

RawPhotoForge is an open-source GPU photo editing project built with Rust, WebGPU, and WebAssembly.

The project focuses on high performance, non-destructive editing, and cross-platform design.

---

## Active Development

- rust
- web

## Archived

- web-ts
- rust-godot-legacy
- python-legacy

---

## Repository Structure

```text
RawPhotoForge/
├── rust/                # Rust wasm core (current main implementation)
├── web/                 # Web UI (current main implementation)
├── web-ts/              # TypeScript-based RawPhotoForge (development ended)
├── rust-godot-legacy/   # Rust + Godot version (development ended)
└── python-legacy/       # Original Python version (development ended)
```

### rust

- Rust image processing core
- GPU image processing using wgpu
- WebAssembly support
- Current main implementation

### web

- TypeScript Web UI
- Uses the Rust wasm core
- Runs in the browser
- Static hosting compatible
- Current main implementation

### web-ts

- WebGPU-based
- TypeScript
- Runs in the browser
- Static hosting
- Development has ended and is no longer maintained

### rust-godot-legacy

- Core: Rust
- GPU: wgpu
- UI: Godot Engine + GDExtension
- RAW image support
- Development has ended and is no longer maintained

### python-legacy

- Initial implementation
- Python-based
- Preserved for historical releases
- Development has ended and is no longer maintained

---

## License

### Project License

This project is licensed under the

**GNU Affero General Public License v3.0 (AGPL-3.0)**

See the `LICENSE` file for details.

### Third-Party Dependency Licenses (Rust)

A list of licenses for Rust dependencies can be found in:

- `rust_licenses.html`

This file is generated using `cargo-about`.

Example:

```bash
cargo about generate about.hbs > rust_licenses.html
```

---

## AI Tools Used

The following AI tools are used to assist development:

- ChatGPT
- Gemini

---

## Philosophy

- Open source
- Local-first
- Cross platform
- GPU Vendor-free
- GPU accelerated
- Non-destructive editing

RawPhotoForge aims to provide a fast photo editing experience that does not depend on a specific operating system or GPU vendor.

---

## Notes

- The project is being developed as a high-quality GPU photo editor
- The design and implementation are continuously evolving

---