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

## 4. 依存関係のexiftoolをダウンロード
[exiftool](https://exiftool.org/)をダウンロードして環境変数PATHに登録 (コマンドが実行できるようにする)


をダウンロードしてRawPhotoForge ディレクトリ内に配置 ("RawPhotoForge\sam2.1_hiera_l.yaml", "RawPhotoForge\sam2.1_hiera_large.pt")


## 5. ライブラリをインストール
```bash
pip install -r raw_photo_forge\requirements.txt
```

## 6. 実行
```bash
python raw_photo_forge\raw_photo_forge.py
```



