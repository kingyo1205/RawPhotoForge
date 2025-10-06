# RawPhotoForge

RAW Photo Editor Written in Python

**This README is available in English and Japanese.**
**このREADMEは英語版と日本語版があります。**

* [English README](#english-readme)
* [日本語 README](#日本語-readme)

---

# English README

RAW Photo Editor Written in Python

---

**Simple yet Powerful RAW Photo Editor**
RawPhotoForge is a **RAW photo editor written in Python**.
With a fast image processing engine that runs on **OpenCL**, you can edit RAW photos with real-time preview.
You can finely adjust **color and brightness using tone curves** for brightness, hue, saturation, and luminance.
It also supports **AI masks (SAM) for partial adjustments**, lens correction via Lensfun, and metadata display using ExifTool.

---

## Supported OS

* **Windows**
* **Linux**

## Supported languages (UI)

* English
* Japanese

---

## Note

All command examples in this README are written for Windows. If you use Linux, please modify commands and paths accordingly.
This software is **distributed as source code only**. Python 3.11 or higher is required to run it.

---

# Usage

## 1. Get the source code

**Be sure to download the custom Asset (zip file).**

* File name format:
  `RawPhotoForge-v<version>.zip`
  Example: `RawPhotoForge-v0.1.0.zip`

* The latest release can be downloaded directly here:
  [Download Latest](https://github.com/kingyo1205/RawPhotoForge/releases/download/v0.2.0/RawPhotoForge-v0.2.0.zip)

※ Do **NOT** use GitHub’s auto-generated “Source code (zip/tar.gz)”. Always use the custom Asset above.

Please download the **zip file** from the GitHub Release page.
Since download counts are tracked, we recommend **using release downloads instead of git clone**.

1. Download and extract the zip file, place the `RawPhotoForge` folder anywhere
2. Change your working directory to `RawPhotoForge`:

```bash
cd RawPhotoForge
```

---

## 2. Place the AI model

Download the following files from [sam2.1-hiera-large page](https://huggingface.co/facebook/sam2.1-hiera-large/tree/main) and place them **in the RawPhotoForge folder root**:

* `sam2.1_hiera_large.pt`
* `sam2.1_hiera_l.yaml`

Example: `RawPhotoForge\sam2.1_hiera_large.pt`

---

## 3. Install ExifTool

Download ExifTool from the [official site](https://exiftool.org/) and add it to the `PATH` environment variable.

**Verification (Windows):**

```bash
exiftool -ver
```

* If the version number is displayed → OK
* If not, check your PATH settings

---

## 4. Install dependencies

```bash
pip install -r raw_photo_forge\requirements.txt
```

* Installs all required Python libraries
* If you encounter errors, check your Python installation and version

---

## 5. Launch RawPhotoForge

```bash
python raw_photo_forge\raw_photo_forge.py
```

* The software will start
* If it fails, check your Python setup and ExifTool PATH

---

# About AI-generated Code

Some parts of this repository were generated/assisted using ChatGPT, Claude, Gemini CLI, and Poe.
No code from projects that cannot be released under OSS (e.g. LMArena) is included.
The `index.html` page was generated using Claude.

---

# License

This repository is distributed under the **GNU AGPLv3**.

## Dependent AI Model

* [sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large/tree/main) (Apache 2.0)

## Dependency: ExifTool

This software uses ExifTool by Phil Harvey.
ExifTool is licensed under the Artistic License (not GPL, Artistic License chosen).
Details: [Artistic License official page](https://dev.perl.org/licenses/artistic.html)

## Dependencies and Licenses

(Checked in 2025 / based on PyPI info)

| Library                                                    | License                            | Notes         |
| ---------------------------------------------------------- | ---------------------------------- | ------------- |
| [numpy](https://pypi.org/project/numpy/)                   | BSD                                | -             |
| [pyopencl](https://pypi.org/project/pyopencl/)             | MIT                                | -             |
| [pillow](https://pypi.org/project/Pillow/)                 | MIT-CMU                            | -             |
| [opencv-python](https://pypi.org/project/opencv-python/)   | Apache-2.0                         | -             |
| [scipy](https://pypi.org/project/scipy/)                   | BSD                                | -             |
| [numba](https://pypi.org/project/numba/)                   | BSD                                | -             |
| [lensfunpy](https://pypi.org/project/lensfunpy/)           | MIT                                | -             |
| [rawpy](https://pypi.org/project/rawpy/)                   | MIT                                | -             |
| [torch](https://pypi.org/project/torch/)                   | BSD                                | -             |
| [sam2](https://pypi.org/project/sam2/)                     | Apache-2.0                         | -             |
| [matplotlib](https://pypi.org/project/matplotlib/)         | Python Software Foundation License | -             |
| [photo-metadata](https://pypi.org/project/photo-metadata/) | MIT                                | Custom (self) |

---


# 日本語 README

# **RawPhotoForge**

RAW Photo Editor Written in Python

---
**シンプルで高機能なRAW現像ソフトウェア**   
RawPhotoForgeは**Pythonで書かれた**RAW現像ソフトです。  
**OpenClで動作**する高速な画像処理エンジンにより、RAW写真をリアルタイムでプレビューしながら編集できます。  
明るさ、色相、彩度、輝度のトーンカーブで、**色や明るさを細かく調整可能**です。  
**AIマスク（SAM）による部分補正**、Lensfun によるレンズ補正、ExifTool によるメタデータ表示もサポートしています。

---

## 対応OS

- **Windows**
- **Linux**

## Supported languages (UI)

- English
- Japanese


## 注意
本 README のコマンド例は Windows 用に書かれています。Linux で使用する場合はコマンドやパスを適宜変更してください。  
本ソフトウェアは **ソースコード配布のみ** です。実行には Python 3.11 以上が必要です。

---

# 使い方

## 1. ソースコードを取得


**必ず自作Asset（zipファイル）をダウンロードしてください。**

- ファイル名フォーマット：  
  `RawPhotoForge-v<バージョン>.zip`  
  例：`RawPhotoForge-v0.1.0.zip`

- 最新リリースはこちらから直接ダウンロード可能：  
  [最新版ダウンロード](https://github.com/kingyo1205/RawPhotoForge/releases/download/v0.2.0/RawPhotoForge-v0.2.0.zip)

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

例: `RawPhotoForge\sam2.1_hiera_large.pt`

---

## 3. ExifTool の導入

[ExifTool 公式サイト](https://exiftool.org/) からダウンロードして、環境変数 `PATH` に追加してください。

**確認手順 (Windows):**

```bash
exiftool -ver
```

* バージョン番号が表示されれば OK
* 表示されない場合は、PATH 設定を確認してください

---

## 4. 依存ライブラリのインストール

```bash
pip install -r raw_photo_forge\requirements.txt
```

* Python ライブラリを一括でインストール
* エラーが出る場合は Python のインストールやバージョンを確認してください

---

## 5. RawPhotoForge を起動

```bash
python raw_photo_forge\raw_photo_forge.py
```

* ソフトが起動します
* 起動できない場合は Python や ExifTool の PATH 設定を確認してください

---

# AIによるコード生成について

本リポジトリの一部コードは ChatGPT, Claude, Gemini CLI, Poe を用いて生成・補助しました。
LMArena 等の OSS にできない可能性がある生成物は一切含まれていません。
index.html (ページ)はClaudeで生成しました。

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

| ライブラリ                                                      | ライセンス       | 備考      |
| ---------------------------------------------------------- | ----------- | ------- |
| [numpy](https://pypi.org/project/numpy/)                   | BSD         | -       |
| [pyopencl](https://pypi.org/project/pyopencl/)             | MIT         | -       |
| [pillow](https://pypi.org/project/Pillow/)                 | MIT-CMU     | -       |
| [opencv-python](https://pypi.org/project/opencv-python/)   | Apache-2.0  | -       |
| [scipy](https://pypi.org/project/scipy/)                   | BSD         | -       |
| [numba](https://pypi.org/project/numba/)                   | BSD         | -       |
| [lensfunpy](https://pypi.org/project/lensfunpy/)           | MIT         | -       |
| [rawpy](https://pypi.org/project/rawpy/)                   | MIT         | -       |
| [torch](https://pypi.org/project/torch/)                   | BSD         | -       |
| [sam2](https://pypi.org/project/sam2/)                     | Apache 2.0  | -       |
| [matplotlib](https://pypi.org/project/matplotlib/)         | Python Software Foundation License | -       |
| [photo-metadata](https://pypi.org/project/photo-metadata/) | MIT         | 自作ライブラリ |


---
