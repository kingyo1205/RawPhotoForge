# RawPhotoForge

RAW Photo Editor Written in Python  

RawPhotoForge is a **RAW photo editor written in Python**.
With a fast image processing engine that runs on **slang (slangpy)**, you can edit RAW photos with real-time preview.
You can finely adjust **color and brightness using tone curves** for brightness, hue, saturation, and luminance.
It also supports **AI masks (SAM) for partial adjustments**, lens correction via Lensfun, and metadata display using ExifTool.

---

> 📕 [README_Japanese](https://github.com/kingyo1205/RawPhotoForge/blob/main/README_Japanese.md)

---

**A simple and powerful RAW development software**
RawPhotoForge is a **RAW photo editor written in Python**.
With a fast image-processing engine powered by **slang (slangpy)**, you can edit RAW photos with **real-time preview**.
Brightness, hue, saturation, and luminance tone curves allow **fine-grained color and light adjustments**.
It also supports **AI masking (SAM)** for partial corrections, lens correction via Lensfun, and metadata display via ExifTool.

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
If you are using Windows, please adjust paths and commands accordingly.
This software is distributed as **source code only**.
To run it, you need **Python 3.11 or later** and an environment where **slang (slangpy) is available**.

---

# Usage

## 1. Download the Source Code

**Be sure to download the custom asset (zip file)**.

* File name format:
  `RawPhotoForge-v<version>.zip`
  Example: `RawPhotoForge-v0.4.0.zip`

* Download from the release page:
  [GitHub Releases](https://github.com/kingyo1205/RawPhotoForge/releases)

※ Do **not** use GitHub’s auto-generated “Source code (zip/tar.gz)”. Always use the custom asset above.

Please download the source code as a **zip file** from the GitHub release page.
To track download counts, using **releases instead of git clone** is recommended.

1. Download and extract the zip file, then place the `RawPhotoForge` folder anywhere you like
2. Move to the directory:

```bash
cd RawPhotoForge
```

---

## 2. Place the AI Model

Download the following files from the
[sam2.1-hiera-large download page](https://huggingface.co/facebook/sam2.1-hiera-large/tree/main)
and place them **directly inside the RawPhotoForge folder**.

* `sam2.1_hiera_large.pt`
* `sam2.1_hiera_l.yaml`

Example: `./sam2.1_hiera_large.pt`

---

## 3. Install ExifTool

Download ExifTool from the
[official website](https://exiftool.org/)
and add it to your system `PATH`.

**Check installation:**

```bash
exiftool -ver
```

* If a version number appears, it’s installed correctly
* If not, verify your PATH settings

---

## 4. Install Dependencies

```bash
pip install -r raw_photo_forge/requirements.txt
```

* Installs required Python libraries
* If errors occur, check your Python version and environment

---

## 5. Launch RawPhotoForge

```bash
python raw_photo_forge/raw_photo_forge.py
```

* The software will start

---

# About AI-generated Code

Some of the code in this repository was generated or assisted using
ChatGPT, Claude, Gemini CLI, and Poe.
No output that would prevent OSS licensing (such as LMArena-incompatible outputs) is included.
`index.html` (the page) was generated using Claude and Gemini CLI.

---

# License

This repository is distributed under **GNU AGPLv3**.

## Included AI Model

* [sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large/tree/main) (Apache 2.0)

## ExifTool

This software uses ExifTool by Phil Harvey.
ExifTool is licensed under the Artistic License (not GPL).
See the official page for details:
[https://dev.perl.org/licenses/artistic.html](https://dev.perl.org/licenses/artistic.html)

## Dependencies and Licenses

(Checked in 2025 / based on PyPI)

| Library        | License                            | Notes          |
| -------------- | ---------------------------------- | -------------- |
| numpy          | BSD                                | -              |
| slangpy        | Apache-2.0                         | -              |
| pillow         | MIT-CMU                            | -              |
| opencv-python  | Apache-2.0                         | -              |
| scipy          | BSD                                | -              |
| lensfunpy      | MIT                                | -              |
| rawpy          | MIT                                | -              |
| torch          | BSD                                | -              |
| sam2           | Apache-2.0                         | -              |
| matplotlib     | Python Software Foundation License | -              |
| photo-metadata | MIT                                | Custom library |

---

# If you like this software, please consider giving it a ⭐ on GitHub!

---

