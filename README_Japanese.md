

# RawPhotoForge
**高機能 & 高速 & 非破壊 RAW現像ソフトウェア**

![RawPhotoForge Screenshot](images/screenshot-main.png)

RawPhotoForgeはローカル環境で動作するRAW現像ソフトです。
初期はPythonで実装していましたが、現在は **Rust版が最新かつメインの実装** です。

Rust版では **wgpu** を中心に設計しており、CUDAなどのベンダー依存GPU APIは使用していません。
移植性、性能、将来の拡張性を重視しています。

---

## リポジトリ構成

```

RawPhotoForge/
├── python-legacy/   # 旧Python版 (v0.5.0まで)
└── rust/            # 現行Rust版 (開発中)

```

### python-legacy
- 初期実装
- Pythonベース
- 既存リリースの保存用
- **現在は保守していません**

### rust (最新)
- コア: Rust
- GPU: wgpu (ベンダー非依存)
- UI: Godot Engine + GDExtension


---

## Rust版の構成


```
rust/
├── photo-editor           # 画像処理コア
├── photo-editor-godot     # Godot用GDExtension
└── raw-photo-forge        # Godotプロジェクト(UI)
```

- `photo-editor`
  - RAW画像処理
  - GPU画像演算
  - メタデータ処理
- `photo-editor-godot`
  - RustコアとGodotの接続層
- `raw-photo-forge`
  - UI
  - プレビュー
  - 各種設定

---

## ライセンス

### プロジェクト本体
本プロジェクトは **GNU Affero General Public License v3.0 (AGPL-3.0)** で公開しています。  
詳細は `LICENSE` ファイルを参照してください。

### 依存関係のライセンス (Rust)
Rust版で使用している依存関係のライセンス一覧は以下にまとめています。

- **`rust_licenses.html`**

このファイルは `cargo-about` を使用して生成しています。

例:
```bash
cargo about generate about.hbs > rust_licenses.html
```

---

## 使用しているAIツール
RawPhotoForgeの開発では、以下のAIツールを活用しています。

- ChatGPT
- Gemini
- Gemini CLI
- Claude

---

## Godot Engineについて

* UIは **Godot Engine** を使用
* Rustとは **GDExtension** で連携
* Godot Engine自体は MIT License で提供されています
  詳細はGodot公式サイトを参照してください

---

## 注意事項


* 商用RAW現像ソフトの代替を目標に開発中
* 設計および実装は継続的に改善中

---

## 開発状況

* Python版: 開発終了
* Rust版: **開発継続中(最新)**


