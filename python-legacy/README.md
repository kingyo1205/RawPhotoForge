# RawPhotoForge

RAW Photo Editor Written in Python  

RawPhotoForge is a **RAW photo editor written in Python**.
With a fast image processing engine that runs on **slang (slangpy)**, you can edit RAW photos with real-time preview.
You can finely adjust **color and brightness using tone curves** for brightness, hue, saturation, and luminance.
It also supports **AI masks (SAM) for partial adjustments**, lens correction via Lensfun, and metadata display using ExifTool.

---

> 📕 [README_Japanese](https://github.com/kingyo1205/RawPhotoForge/blob/main/python-legacy/README_Japanese.md)

---

**Simple and powerful RAW development software**
RawPhotoForge is a RAW photo editor **written in Python**.
With its high-speed image processing engine powered by **slang (slangpy)**, you can edit RAW photos with **real-time preview**.
You can finely adjust **brightness, hue, saturation, and luminance tone curves**.
It also supports **AI masking (SAM) for selective adjustments**, lens correction via Lensfun, and metadata display via ExifTool.

---

## Supported OS

* **Windows**
* **Linux**

## Supported Languages (UI)

* English
* Japanese

---

### Note

All command examples in this README are for **Linux environments**.
If you are using Windows, please adapt the commands and paths accordingly.
This software is distributed **as source code only**.
To run it, you need **Python 3.11 or later** and an environment where **slang (slangpy)** is available.

---

# Usage

## 1. Obtain the source code

**Be sure to download the custom Asset (zip file) created by the author.**

* File name format:
  `RawPhotoForge-v<version>.zip`
  Example: `RawPhotoForge-v0.5.0.zip`

* Download from the Release page:
  [GitHub Releases](https://github.com/kingyo1205/RawPhotoForge/releases)

※ Do **not** use GitHub’s auto-generated “Source code (zip/tar.gz)”.
Please make sure to use the custom Asset above.

Download the zip file from the GitHub release page, extract it, and place the `RawPhotoForge` folder anywhere you like.
Since I want to count downloads, **using the release zip instead of git clone is recommended**.

1. Download and extract the zip file, then place the `RawPhotoForge` folder anywhere.
2. Move into the `RawPhotoForge` directory:

```bash
cd RawPhotoForge
```

---

## 2. Place the AI model files

Download the following files from
[sam2.1-hiera-large download page](https://huggingface.co/facebook/sam2.1-hiera-large/tree/main)
and place them **directly under the RawPhotoForge folder**.

* `sam2.1_hiera_large.pt`
* `sam2.1_hiera_l.yaml`

Example: `./sam2.1_hiera_large.pt`

---

## 3. Install ExifTool

Download ExifTool from the [official website](https://exiftool.org/) and add it to your `PATH` environment variable.

**Verification:**

```bash
exiftool -ver
```

* If a version number appears, everything is OK.
* If nothing appears, check your PATH settings.

---

## 4. Install dependency libraries

```bash
pip install -r raw_photo_forge/requirements.txt
```

* Installs all required Python libraries.
* If errors occur, check your Python installation or version.

---

## 5. Launch RawPhotoForge

```bash
python raw_photo_forge/raw_photo_forge.py
```

* The software will start.

---

# About AI-generated code

Some parts of this repository were created or assisted using ChatGPT, Claude, Gemini CLI, and Poe.
No content that would prevent it from being open-source (e.g., content restricted by LMArena rules) is included.
`index.html` (the webpage) was generated using Claude and Gemini CLI.

---

# License

This repository is distributed under the **GNU AGPLv3** license.

## Dependent AI model

* [sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large/tree/main) (Apache 2.0)

## ExifTool Dependency

This software uses ExifTool by Phil Harvey.

ExifTool is dual-licensed under GPL 1.0 or later, or Artistic License 1.0. To ensure consistency with AGPL, this software utilizes ExifTool under the terms of the **GNU General Public License (GPL) version 3**.

See the [official ExifTool page](https://exiftool.org/) for details.

## Dependency libraries and licenses

(Confirmed in 2025 / based on PyPI information)

| Library                                                    | License                            | Notes          |
| ---------------------------------------------------------- | ---------------------------------- | -------------- |
| [numpy](https://pypi.org/project/numpy/)                   | BSD                                | -              |
| [slangpy](https://pypi.org/project/slangpy/)               | Apache-2.0 WITH LLVM-exception     | -              |
| [pillow](https://pypi.org/project/Pillow/)                 | MIT-CMU                            | -              |
| [opencv-python](https://pypi.org/project/opencv-python/)   | Apache-2.0                         | -              |
| [scipy](https://pypi.org/project/scipy/)                   | BSD                                | -              |
| [lensfunpy](https://pypi.org/project/lensfunpy/)           | MIT                                | -              |
| [rawpy](https://pypi.org/project/rawpy/)                   | MIT                                | -              |
| [torch](https://pypi.org/project/torch/)                   | BSD                                | -              |
| [sam2](https://pypi.org/project/sam2/)                     | Apache 2.0                         | -              |
| [matplotlib](https://pypi.org/project/matplotlib/)         | Python Software Foundation License | -              |
| [photo-metadata](https://pypi.org/project/photo-metadata/) | MIT                                | Custom library |

---

# If you like this software, please consider giving it a ⭐ on GitHub!

---


