
# RawPhotoForge
Python で書かれた 高速 RAW 現像ソフトウェア

RawPhotoForge はシンプルで高機能な RAW 現像ソフトです。  
OpenCL と NumPy バックエンド対応の高速画像処理エンジンで、RAW 写真をリアルタイムにプレビューしながら編集できます。  
明るさ、色相、彩度、輝度のトーンカーブで色や輝度を細かく調整可能です。  
さらに、AI マスク（SAM）による部分補正、Lensfun によるレンズ補正、ExifTool によるメタデータ表示もサポートしています。

---

## 対応 OS

- Windows
- Linux

**注意:**

- 本 README のコマンド例は Windows 向けです。Linux では適宜コマンドやパスを変更してください
- 本ソフトウェアは **ソースコード配布のみ** です。Python 3.11 以上が必要です

---

# 使い方

## 1. ソースコードを取得

ソースコードは GitHub のリリースページから **zip ファイル** をダウンロードしてください。  
ダウンロード数を集計したいため、可能な限り **git clone ではなくリリースから取得** を推奨します。

[Releases ページ](https://github.com/kingyo1205/RawPhotoForge/releases)

**手順:**

1. zip ファイルをダウンロードして解凍し、任意の場所に `RawPhotoForge` フォルダを置きます
2. カレントディレクトリを `RawPhotoForge` に移動します

```bash
cd RawPhotoForge
````

> ※ git clone で取得する場合も同様にカレントディレクトリを移動してください

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

# AI によるコード生成について

本リポジトリの一部コードは ChatGPT、Claude、Gemini CLI、Poe を用いて生成・補助しました。
LMArena 等の OSS にできない可能性がある生成物は一切含まれていません。

---

# ライセンス

このリポジトリは **MIT License** で配布しています。

## 依存 AI モデル

* [sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large/tree/main) (Apache 2.0)

## 依存関係の ExifTool

ExifTool は Phil Harvey 氏によるツールで、**Artistic License** に基づいています。
詳細は [公式ページ](https://dev.perl.org/licenses/artistic.html) をご覧ください

---

## 主要な依存ライブラリとライセンス

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
| [matplotlib](https://pypi.org/project/matplotlib/)         | PSF License | -       |
| [photo-metadata](https://pypi.org/project/photo-metadata/) | MIT         | 自作ライブラリ |



