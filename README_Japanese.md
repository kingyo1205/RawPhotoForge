# RawPhotoForge

RAW Photo Editor Written in Python

---
**シンプルで高機能なRAW現像ソフトウェア**   
RawPhotoForgeは**Pythonで書かれた**RAW現像ソフトです。  
**slang (slangpy) で動作**する高速な画像処理エンジンにより、RAW写真をリアルタイムでプレビューしながら編集できます。  
明るさ、色相、彩度、輝度のトーンカーブで、**色や明るさを細かく調整可能**です。  
**AIマスク（SAM）による部分補正**、Lensfun によるレンズ補正、ExifTool によるメタデータ表示もサポートしています。

---

## 対応OS

- **Windows**
- **Linux**

## 対応言語 (UI)

- English
- Japanese

---

### 注意

本 README に記載しているコマンド例は **Liunx 環境向け** です。Windows をご利用の場合は、コマンドやパスを適宜読み替えてください。
本ソフトウェアは **ソースコードのみを配布** しています。実行には **Python 3.11 以上** および **slang (slangpy) が利用可能な環境** が必要です。


---

# 使い方

## 1. ソースコードを取得


**必ず自作Asset（zipファイル）をダウンロードしてください。**

- ファイル名フォーマット：  
  `RawPhotoForge-v<バージョン>.zip`  
  例：`RawPhotoForge-v0.4.0.zip`

- リリースページからダウンロード：  
  [GitHub Releases](https://github.com/kingyo1205/RawPhotoForge/releases)

※ GitHubの自動生成「Source code (zip/tar.gz)」ではなく、必ず上記の自作Assetを使用してください。

ソースコードは GitHub のリリースページから **zip ファイル** をダウンロードしてください。  
ダウンロード数を集計したいため、可能な限り **git clone ではなくリリースから取得** を推奨します。


1. zip ファイルをダウンロードして解凍し、任意の場所に `RawPhotoForge` フォルダを置きます
2. カレントディレクトリを `RawPhotoForge` に移動します：

```bash
cd RawPhotoForge
```

---

## 2. AI モデルを配置

[sam2.1-hiera-large ダウンロードページ](https://huggingface.co/facebook/sam2.1-hiera-large/tree/main) から以下のファイルを取得し、**RawPhotoForge フォルダ直下**に置きます。

* `sam2.1_hiera_large.pt`
* `sam2.1_hiera_l.yaml`

例: `./sam2.1_hiera_large.pt`

---

## 3. ExifTool の導入

[ExifTool 公式サイト](https://exiftool.org/) からダウンロードして、環境変数 `PATH` に追加してください。

**確認手順:**

```bash
exiftool -ver
```

* バージョン番号が表示されれば OK
* 表示されない場合は、PATH 設定を確認してください

---

## 4. 依存ライブラリのインストール

```bash
pip install -r raw_photo_forge/requirements.txt
```

* Python ライブラリを一括でインストール
* エラーが出る場合は Python のインストールやバージョンを確認してください

---

## 5. RawPhotoForge を起動

```bash
python raw_photo_forge/raw_photo_forge.py
```

* ソフトが起動します

---

# AIによるコード生成について

本リポジトリの一部コードは ChatGPT, Claude, Gemini CLI, Poe を用いて生成・補助しました。
LMArena 等の OSS にできない可能性がある生成物は一切含まれていません。
index.html (ページ) はClaude, Gemini CLIで生成しました。

---

# ライセンス

このリポジトリは **GNU AGPLv3** で配布しています。

## 依存 AI モデル

* [sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large/tree/main) (Apache 2.0)

## 依存関係の ExifTool

このソフトウェアでは Phil Harvey 氏による ExifTool を使用しています。
ExifTool は Artistic License に基づいてライセンスされています（GPL ではなく Artistic License を選択）。
詳細は [公式ページ](https://dev.perl.org/licenses/artistic.html) をご覧ください。

## 依存ライブラリとライセンス

（2025年確認 / PyPI 記載情報）

| ライブラリ                                                     | ライセンス                         | 備考           |
| -------------------------------------------------------------- | ---------------------------------- | -------------- |
| [numpy](https://pypi.org/project/numpy/)                       | BSD                                | -              |
| [slangpy](https://pypi.org/project/slangpy/)                   | Apache-2.0                         | -              |
| [pillow](https://pypi.org/project/Pillow/)                     | MIT-CMU                            | -              |
| [opencv-python](https://pypi.org/project/opencv-python/)       | Apache-2.0                         | -              |
| [scipy](https://pypi.org/project/scipy/)                       | BSD                                | -              |
| [lensfunpy](https://pypi.org/project/lensfunpy/)               | MIT                                | -              |
| [rawpy](https://pypi.org/project/rawpy/)                       | MIT                                | -              |
| [torch](https://pypi.org/project/torch/)                       | BSD                                | -              |
| [sam2](https://pypi.org/project/sam2/)                         | Apache 2.0                         | -              |
| [matplotlib](https://pypi.org/project/matplotlib/)             | Python Software Foundation License | -              |
| [photo-metadata](https://pypi.org/project/photo-metadata/)     | MIT                                | 自作ライブラリ |


---
# このソフトウェアが気に入ったら、ぜひGitHubで⭐をお願いします！
---


