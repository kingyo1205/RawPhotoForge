# RawPhotoForge
RAW Photo Editor Written in Python


# 使い方

## 1. git clone

```bash
git clone https://github.com/kingyo1205/RawPhotoForge.git
```

## 2. カレントディレクトリにする
```bash
cd RawPhotoForge
```

## 3. 依存関係のAIモデルをダウンロード
[sam2.1-hiera-largeのダウンロードページ](https://huggingface.co/facebook/sam2.1-hiera-large/tree/main)から
- sam2.1_hiera_large.pt
- sam2.1_hiera_l.yaml

をダウンロードして`RawPhotoForge`ディレクトリ内に配置 (`RawPhotoForge\sam2.1_hiera_l.yaml`, `RawPhotoForge\sam2.1_hiera_large.pt`)

## 4. 依存関係のexiftoolをダウンロード
[exiftool](https://exiftool.org/)をダウンロードして環境変数`PATH`に登録 (コマンドが実行できるようにする)



## 5. ライブラリをインストール
```bash
pip install -r raw_photo_forge\requirements.txt
```

## 6. 実行
```bash
python raw_photo_forge\raw_photo_forge.py
```

# ビルド方法 (exe化)

## 1. pyinstallerをインストール
```bash
pip install pyinstaller
```

## 2. ビルド
```bash
pyinstaller --collect-all raw_image_editor --collect-all numpy --collect-all pyopencl --collect-all pillow --collect-all opencv-python --collect-all scipy --collect-all numba --collect-all lensfunpy --collect-all rawpy --collect-all torch --collect-all sam2 --collect-all matplotlib --collect-all photo-metadata --onedir --add-data "sam2.1_hiera_large.pt;raw_image_editor" --add-data "sam2.1_hiera_l.yaml;raw_image_editor" raw_photo_forge\raw_photo_forge.py
```

## 3. exiftoolを配置
`dist\raw_photo_forge`に`exiftool_dir`を作成して、`exiftool.exe`を配置。こうなるように`dist\raw_photo_forgeにexiftool_dir\exiftool.exe`




# AIによるコード生成について
本リポジトリの一部コードは ChatGPT, Claude, Gemini CLI, Poe を用いて生成・補助しました。  
LMArena等のOSSにできない可能性がある生成物は一切含まれていません。

 

# ライセンス
このリポジトリは**MIT License** の下で配布しています。  

## 依存AIモデル
- [sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large/tree/main) (Apache 2.0)
## 依存ライブラリとそのライセンス

本ソフトウェアは複数のライブラリに依存しています。  
以下は主要ライブラリとライセンスの一覧です（2025年確認）。  
ライセンスはpypiに書いてあるものです。  

| ライブラリ | ライセンス | 備考 |
|------------|------------|------|
| [numpy](https://pypi.org/project/numpy/) | BSD | - |
| [pyopencl](https://pypi.org/project/pyopencl/) | MIT | - |
| [pillow](https://pypi.org/project/Pillow/) | MIT-CMU | - |
| [opencv-python](https://pypi.org/project/opencv-python/) | Apache-2.0 | - |
| [scipy](https://pypi.org/project/scipy/) | BSD | - |
| [numba](https://pypi.org/project/numba/) | BSD | - |
| [lensfunpy](https://pypi.org/project/lensfunpy/) | MIT | - |
| [rawpy](https://pypi.org/project/rawpy/) | MIT | - |
| [torch](https://pypi.org/project/torch/) | BSD | - |
| [sam2](https://pypi.org/project/sam2/) | Apache 2.0  | - |
| [matplotlib](https://pypi.org/project/matplotlib/) | Python Software Foundation License | - |
| [photo-metadata](https://pypi.org/project/photo-metadata/) | MIT | 私の自作ライブラリ |










