# RawPhotoForge

RAW Photo Editor Written in Python

シンプルで高機能な **RAW現像ソフトウェア**   
RawPhotoForgeはPythonで書かれたRAW現像ソフトです。  
OpenClとNumPyバックエンド対応で高速な画像処理エンジンにより、RAW写真をリアルタイムでプレビューしながら編集できます。  
明るさ、色相、彩度、輝度のトーンカーブで、色や明るさを細かく調整可能です。
AI マスク（SAM）による部分補正、Lensfun によるレンズ補正、ExifTool によるメタデータ表示もサポートしています。



**対応OS:** **Windows**, **Linux** 

**注意:** 本 README のコマンド例は Windows 用に書かれています。  
Linux で使用する場合はコマンドやパスを適宜変更してください。


本ソフトウェアは **ソースコード配布のみ** です。  
実行には Python 3.11 以上が必要です。

---

# 使い方

## 1. ソースコードを取得

ソースコードは GitHub のリリースページから **zip ファイル** をダウンロードしてください。  
ダウンロード数を集計したいため、可能な限り **git clone ではなくリリースから取得** を推奨します。

[Releases ページ](https://github.com/kingyo1205/RawPhotoForge/releases)

1. zip ファイルをダウンロードして解凍し、任意の場所に `RawPhotoForge` フォルダを置きます
2. カレントディレクトリを `RawPhotoForge` に移動します：

```bash
cd RawPhotoForge
```

---

## 2. AIモデルを配置

[sam2.1-hiera-large のダウンロードページ](https://huggingface.co/facebook/sam2.1-hiera-large/tree/main)から以下を取得し、**RawPhotoForge フォルダ直下**に置いてください:

* `sam2.1_hiera_large.pt`
* `sam2.1_hiera_l.yaml`

> こんな感じです。`RawPhotoForge\sam2.1_hiera_large.pt`

---

## 3. ExifTool の導入

[ExifTool 公式サイト](https://exiftool.org/)からダウンロードして、環境変数 `PATH` に登録してください。

* Windows の場合、「環境変数を編集」で ExifTool のフォルダを追加
* コマンドプロンプトで以下を入力して確認できます：

```bash
exiftool -ver
```

バージョン番号が表示されれば OK です

---

## 4. 依存ライブラリのインストール

```bash
pip install -r raw_photo_forge\requirements.txt
```

* Python のライブラリを一括でインストールします
* エラーが出た場合は、Python が正しくインストールされているか確認してください

---

## 5. RawPhotoForge を起動

```bash
python raw_photo_forge\raw_photo_forge.py
```

* これでソフトが起動します
* もし「コマンドが見つかりません」などのエラーが出た場合は、Python のパスや ExifTool の PATH 設定を確認してください

---

# AIによるコード生成について

本リポジトリの一部コードは ChatGPT, Claude, Gemini CLI, Poe を用いて生成・補助しました。
LMArena 等の OSS にできない可能性がある生成物は一切含まれていません。

---

# ライセンス

このリポジトリは **MIT License** の下で配布しています。

## 依存 AI モデル

* [sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large/tree/main) (Apache 2.0)

## 依存関係の ExifTool

このソフトウェアでは Phil Harvey 氏による ExifTool を使用しています。
ExifTool は Artistic License に基づいてライセンスされています（GPL ではなく Artistic License を選択）。
詳細は [公式ページ](https://dev.perl.org/licenses/artistic.html) をご覧ください。

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

