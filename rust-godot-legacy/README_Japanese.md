# RawPhotoForge Rust+Godot

このディレクトリは旧実装です

このディレクトリには
RawPhotoForgeの旧Rust+Godot版が含まれています。
本ディレクトリはアーカイブ目的で残されています。

## Rust+Godot版の構成

```text
rust-godot-legacy/
├── photo-editor           # 画像処理コア
├── photo-editor-godot     # Godot用GDExtension
└── raw-photo-forge        # Godotプロジェクト(UI)
```

### photo-editor
- RAW画像処理
- GPU画像演算
- メタデータ処理

### photo-editor-godot
- RustコアとGodotの接続層

### raw-photo-forge
- UI
- プレビュー
- 各種設定

---

## 技術スタック

- Rust
- wgpu
- Godot Engine
- GDExtension

---

## Godot Engineについて

- UIは **Godot Engine** を使用
- Rustとは **GDExtension** で連携
- Godot Engine自体は MIT License で提供されています。詳細はGodot公式サイトを参照してください


---

## ビルド方法

RawPhotoForge Desktopは、
Rustで実装されたコアと、
Godot(GDExtension)によるUIで構成されています。

以下はLinux環境でソースからビルドする手順です。

### 必要環境

- Rust
- Cargo
- Godot Engine (通常版 .NET版不要)
- Linux (x86_64で動作確認)

### 手順

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

### Godotでのエクスポート

1. Godotで `rust-godot-legacy/raw-photo-forge/` ディレクトリをプロジェクトとして開く
2. GDExtensionが正しく読み込まれていることを確認
3. Godotのエクスポート機能を使ってバイナリを生成

---


## ライセンス

### プロジェクト本体

本プロジェクトは
**GNU Affero General Public License v3.0 (AGPL-3.0)**
で公開しています。

詳細は `LICENSE` ファイルを参照してください。

### 依存関係のライセンス (Rust)

Rust版で使用している依存関係の
ライセンス一覧は以下にまとめています。

- `rust-godot-legacy/rust_licenses.html`

このファイルは
`cargo-about`
を使用して生成しています。

例:

```bash
cargo about generate about.hbs > rust_licenses.html
```

---

## 動作確認環境

- Linux x86_64

その他の環境は未検証です。
