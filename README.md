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

## 3. 依存関係のaiモデルをダウンロード
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




# AIによるコード生成について
本リポジトリの一部コードは ChatGPT, Claude, Gemini CLI, Poe を用いて生成・補助しました。  
LMArena等のOSSにできない可能性がある生成物は一切含まれていません。

 

# ライセンス
このリポジトリは**MIT License** の下で配布しています。  
## 依存ライブラリとそのライセンス

本ソフトウェアは複数のライブラリに依存しています。  
以下は主要ライブラリとライセンスの一覧です（2025年確認）。

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
| [matplotlib](https://pypi.org/project/matplotlib/) | Python Software Foundation License | - |
| [PySide6](https://pypi.org/project/PySide6/) | LGPL-3.0-only | 差し替え可能。全文はこちら: [https://www.qt.io/licensing/](https://www.qt.io/licensing/) |
| [chardet](https://pypi.org/project/chardet/) | LGPL-2.1+ | 差し替え可能。全文はこちら: [https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html) |
| [tqdm](https://pypi.org/project/tqdm/) | MIT | - |



### LGPL ライブラリについて

PySide6 や chardet のような LGPL ライブラリは、利用者が自由に改変・差し替えできる状態で配布しています。  
Releasesの`pyinstaller`の`--onedir`でビルドしたexeは`--collect-all`オプションを使って LGPL ライブラリのファイルをすべて、exeに埋め込まず、外に出して置き換えられるようにしています。






