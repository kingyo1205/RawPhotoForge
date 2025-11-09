# RawPhotoForge

RAW Photo Editor Written in Python  

RawPhotoForge is a **RAW photo editor written in Python**.
With a fast image processing engine that runs on **WebGPU (wgpu-py)**, you can edit RAW photos with real-time preview.
You can finely adjust **color and brightness using tone curves** for brightness, hue, saturation, and luminance.
It also supports **AI masks (SAM) for partial adjustments**, lens correction via Lensfun, and metadata display using ExifTool.

---

> 📕 [README_Japanese](https://github.com/kingyo1205/RawPhotoForge/blob/main/README_Japanese.md)

---

# English README

RAW Photo Editor Written in Python

---

**Simple yet Powerful RAW Photo Editor**
RawPhotoForge is a **RAW photo editor written in Python**.
With a fast image processing engine that runs on **WebGPU (wgpu-py)**, you can edit RAW photos with real-time preview.
You can finely adjust **color and brightness using tone curves** for brightness, hue, saturation, and luminance.
It also supports **AI masks (SAM) for partial adjustments**, lens correction via Lensfun, and metadata display using ExifTool.

---

## Supported OS

- **Windows**
- **Linux**

## Supported languages (UI)

* English
* Japanese

---

### Note

All command examples in this README are written for **Windows environments**.
If you are using Linux, please adjust the commands and paths accordingly.
This software is distributed as **source code only**. To run it, you need **Python 3.11 or higher** and an environment where **WebGPU (wgpu-py) is available**.


---

# Usage

## 1. Get the source code

**Be sure to download the custom Asset (zip file).**

* File name format:
  `RawPhotoForge-v<version>.zip`  
  Example: `RawPhotoForge-v0.2.0.zip`

* Download from the releases page:
  [GitHub Releases](https://github.com/kingyo1205/RawPhotoForge/releases)

※ Do **NOT** use GitHub's auto-generated "Source code (zip/tar.gz)". Always use the custom Asset above.

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
| [wgpu](https://pypi.org/project/wgpu/)                     | BSD                                | -             |
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
# If you find this software useful, please consider giving it a ⭐ on GitHub!
---


