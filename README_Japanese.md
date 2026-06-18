# RawPhotoForge

GPU Photo Editor Project


### [⭐RawPhotoForgeを試す⭐](https://kingyo1205.github.io/RawPhotoForge/web/dist/index.html)

RawPhotoForgeは
Rust + WebGPU + WebAssemblyで構築された
オープンソースのGPU写真編集プロジェクトです。


高速処理、非破壊編集、クロスプラットフォーム設計を重視しています。

---

## 現在開発中

- rust
- web

## アーカイブ

- web-ts
- rust-godot-legacy
- python-legacy



## リポジトリ構成

```text
RawPhotoForge/
├── rust/             # Rust wasm (現行のメイン)
├── web/              # WebUI (現行のメイン)
├── web-ts/           # Web TypeScript実装のRawPhotoForge (開発終了)
├── rust-godot-legacy/   # Rust+Godot版 (開発終了)
└── python-legacy/    # 旧Python版 (開発終了)
```

### rust
- Rust画像処理コア
- wgpuによるGPU画像処理
- wasm対応
- 現行のメイン

### web
- TypeScript Web UI
- Rust wasmコアを利用
- ブラウザ動作
- 静的ホスティング対応
- 現行のメイン

### web-ts
- WebGPUベース
- TypeScript
- ブラウザ動作
- 静的ホスティング
- 開発終了 現在は保守していません

### rust-godot-legacy
- コア: Rust
- GPU: wgpu
- UI: Godot Engine + GDExtension
- RAW対応
- 開発終了 現在は保守していません

### python-legacy
- 初期実装
- Pythonベース
- 既存リリース保存用
- 開発終了 現在は保守していません

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

- `rust_licenses.html`

このファイルは
`cargo-about`
を使用して生成しています。

例:

```bash
cargo about generate about.hbs > rust_licenses.html
```

---

## 使用しているAIツール

開発補助として以下のAIツールを活用しています。

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

RawPhotoForgeは特定のOSやGPUベンダーに依存せず、
ローカル環境で高速に動作する写真編集体験を目指しています。


---

## 注意事項
- 高品質なGPU写真編集ソフトを目標に開発中
- 設計および実装は継続的に改善中

---

