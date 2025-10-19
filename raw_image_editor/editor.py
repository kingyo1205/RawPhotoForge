# editor.py

import numpy as np
import rawpy
from PIL import Image
from scipy import interpolate
import scipy
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import photo_metadata
import os
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
import torch
from typing import Tuple, Dict, Optional, Union, Any
import lensfunpy
import lensfunpy.util
import sys
import re
import time
import cv2
import wgpu
import copy
from scipy.ndimage import uniform_filter
from scipy.stats import skew, entropy


matplotlib.use('Agg')
if getattr(sys, 'frozen', False):
    WEIGHTS = os.path.join(os.path.dirname(__file__), "sam2.1_hiera_large.pt")
    CONFIG = os.path.join(os.path.dirname(__file__), "sam2.1_hiera_l.yaml")
else:
    WEIGHTS = os.path.join(os.getcwd(), "sam2.1_hiera_large.pt")
    CONFIG = os.path.join(os.getcwd(), "sam2.1_hiera_l.yaml")
DEVICE = "cpu"

sam2 = build_sam2(CONFIG, WEIGHTS, device=DEVICE, apply_postprocessing=True)
predictor = SAM2ImagePredictor(sam2)

class RAWImageEditorBase:
    def __init__(self, file_path: str | None = None, image_array: np.ndarray | None = None, metadata: photo_metadata.Metadata | None = None, exiftool_path: str | None = None, jp_tags_json_path: str | None = None):
        

        if (file_path is None) and (image_array is None) and (metadata is None):
            raise ValueError("You must specify file_path or image_array.")
        
        if (file_path is not None) and (image_array is not None) and (metadata is not None):
            raise ValueError("You cannot specify both image_array and file_path.")
        
        if (file_path is None) and (metadata is not None) and (image_array is None):
            raise ValueError("You cannot use only metadata")
        if file_path is not None:
            self.file_path: str = file_path
            try:
                rawpy_object = rawpy.imread(self.file_path)
                
                self.image_array: np.ndarray = rawpy_object.postprocess(use_camera_wb = True, 
                                                                        output_bps=16,
                                                                        output_color = rawpy.ColorSpace.sRGB,
                                                                        gamma=(1,1), no_auto_bright=True

                                                                        ).astype(np.float32)
                self.is_raw: bool = True
            except rawpy._rawpy.LibRawFileUnsupportedError: # type: ignore
                self.image_array = np.array(Image.open(self.file_path).convert("RGB")).astype(np.float32)
                self.image_array *= 257.0
                self.is_raw: bool = False

            

            if exiftool_path is not None:
                photo_metadata.set_exiftool_path(exiftool_path)
            if jp_tags_json_path is not None:
                photo_metadata.set_jp_tags_json_path(jp_tags_json_path)
            
            self.metadata = photo_metadata.Metadata(file_path)
            self.image_array = (self.image_array / 65535.0)
            self.initial_image_array: np.ndarray = self.image_array.copy()
        

        else:
            self.file_path = None
            self.image_array = image_array
            self.initial_image_array = image_array.copy()
            self.metadata = metadata
            self.is_raw = None

        
        
            
        
        print("\n" + ("-" * 5) + "Image Info" + ("-" * 20))
        print(f"file_path: {self.file_path or 'None'}")
        print(f"image_array shape: {self.image_array.shape}")
        print(f"image_array dtype: {self.image_array.dtype}")
        print(f"image_width: {self.width}")
        print(f"image_height: {self.height}")
        print(f"image_channels: {self.channels}")
        print(f"image_is_raw: {self.is_raw}")
        print("-" * 35)
        

        self.main_adjustments_values: dict[str, Any] = {
            "brightness_tone_curve": np.arange(0, 65536),
            "hue_tone_curve": np.arange(0, 65536),
            "saturation_tone_curve": np.full((65536,), 65535),
            "lightness_tone_curve": np.full((65536,), 65535),
            "wb_temperature": 0,
            "wb_tint": 0,
            "exposure": 0.0,
            "contrast": 0,
            "shadow": 0,
            "highlight": 0,
            "black": 0,
            "white": 0,
            "vignette": 0
        }

        self.mask_adjustments_values: dict[str, dict[str, Any]] = {}

        

        self.p5 = 0.05
        self.p25 = 0.25
        self.p50 = 0.5
        self.p75 = 0.75
        self.p95 = 0.95



        

    
        

    
        
        
    @property
    def width(self) -> int:
        """
        写真の横の長さの値
        """
        
        return self.image_array.shape[1]
    
    @property
    def height(self) -> int:
        """
        写真の縦の長さの値
        """
        return self.image_array.shape[0]
    
    @property
    def channels(self) -> int:
        """
        写真のチャンネル数
        """
        return self.image_array.shape[2]
    
    
    
    def reset(self):
        """
        写真を最初の状態に戻す関数
        """

        self.image_array = self.initial_image_array.copy()
        
        self.normalization()


    def show(self, image: np.ndarray = None):
        """
        写真を表示する関数  

        Parameters:
            image_array (numpy.ndarray, optional): 表示する写真のndarray. Defaults to None. 指定されなかった場合はself.image_arrayを使用する。
            
        """
        
        
            
        if image is None:
            show_image = self.image_array
        else:
            show_image = image
        
        
        Image.fromarray(np.clip(show_image * 255.0, 0.0, 255.0).astype(np.uint8)).show()
        
    
    def save(self, file_path: str, quality: int = 90):
        """
        写真を保存する関数

        exiftoolを使用してメタデータを書き込みます。

        フォーマットはjpeg/png

        Parameters:
            file_path (str): 保存するファイルのパス
            quality (int, optional): 保存する画像の品質 (0〜100). Defaults to 90. 
        """
        
        save_extension = [".jpg", ".jpeg", ".png"]

        if quality < 0 or quality > 100:
            raise ValueError("quality must be between 0 and 100")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in save_extension:
            raise ValueError(f"file_path must be {' or '.join(save_extension)}")
        
        
        
        
        try:
            del self.metadata["EXIF:Orientation"]
        except Exception as e:
            pass
                
        if ext in [".jpg", ".jpeg", ".png"]:
            try:
                # self.image_array = HSVDenoiser(n_segments=2500, percentiles={"H": (20.0, 80.0), "S": (1.0, 99.0), "V": (0.0, 100.0)}).denoise(self.image_array)
                save_image = self.as_uint8()

                
                    
                Image.fromarray(save_image).save(file_path, format=ext[1:].upper(), optimize=True, quality=quality)
                try:
                    self.metadata.write_metadata_to_file(file_path)
                    return "success"
                except Exception as e:
                    print(f"Failed to write metadata to file: {e}")
                    return "metadata_write_error"
            except Exception as e:
                print(f"error: {e}")
                return "error"
        else:
            return "error"
        

    def crop(self, top_left: tuple, bottom_right: tuple, update_initial_image_array: bool = True):
        """
        写真を切り抜く関数
        """
        
        self.image_array = self.image_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
        
        if update_initial_image_array:
            self.initial_image_array = self.image_array.copy()

    
    def as_uint8(self):
        """
        numpyのuint8型の画像を返す関数
        """
        self.normalization()
        return np.clip(self.image_array * 255.0, 0.0, 255.0).astype(np.uint8)
    
    def set_whitebalance_value(self, temperature: int, tint: int, mask_name: np.ndarray = None):
        """
        ホワイトバランスをの調整値を設定する関数

        Parameters:
            temperature (int): 色温度 (-100〜100)
            tint (int): 色かぶり補正 (-100〜100)
            mask_name (np.ndarray, optional): マスク名. Defaults to None.
        """
        if temperature < -100 or temperature > 100:
            raise ValueError("temperature must be between 0 and 100")
        if tint < -100 or tint > 100:
            raise ValueError("tint must be between 0 and 100")
        
        if temperature == 0 and tint == 0:
            return
        if mask_name is None:
            self.main_adjustments_values["wb_temperature"] = temperature
            self.main_adjustments_values["wb_tint"] = tint
        else:
            self.mask_adjustments_values[mask_name]["wb_temperature"] = temperature
            self.mask_adjustments_values[mask_name]["wb_tint"] = tint


    
    
        

    
    def normalization(self):
        """
        写真を値を正規化する関数

        """
        
        
        self.image_array = np.clip(self.image_array, 0.0, 1.0).astype(np.float32)
        
        
        

    
        
        
        

    
    
    
    def lens_correction(self) -> bool:
        """
        lensfunpy を使って self.image_array に対してレンズ補正を行う。
        - 前提:
        - self.image_array: numpy.ndarray, dtype=float32, 値域 0..1, shape (H,W,3), RGB（線形）
        - self.metadata.metadata: dict (exiftool 形式)
        - 返り値: 補正が何か適用されたら True、何もできなければ False
        - 補正種類（検出されたら print する）:
        - ヴィネット (vignetting)
        - サブピクセル幾何歪み（TCA を含む）
        - 通常幾何歪み（ジオメトリ）
        - scale / projection が modifier により設定されている場合は info を出す
        """
        
        

        # OpenCV はあれば高速フォールバックに使う（無くても lensfunpy.util.remap を試す）
        

        # ---- メタデータ取得ユーティリティ ----
        if not hasattr(self, "metadata") or not hasattr(self.metadata, "metadata"):
            print("メタデータが見つかりません。self.metadata.metadata を確認してください。")
            return False
        tags = self.metadata.metadata

        def _first_of(keys):
            for k in keys:
                if k in tags and tags[k] not in (None, ""):
                    return tags[k]
            return None

        def _parse_number(val):
            if val is None:
                return None
            if isinstance(val, (list, tuple)):
                val = val[0]
            if isinstance(val, (bytes, bytearray)):
                try:
                    val = val.decode("utf-8", errors="ignore")
                except:
                    val = str(val)
            if isinstance(val, str):
                # '55/1' や '55 mm' などに対応
                if "/" in val:
                    try:
                        a, b = val.split("/", 1)
                        return float(a) / float(b)
                    except Exception:
                        pass
                m = re.search(r"([0-9]+(?:\.[0-9]+)?)", val)
                if m:
                    try:
                        return float(m.group(1))
                    except:
                        return None
                try:
                    return float(val)
                except:
                    return None
            if isinstance(val, (int, float)):
                return float(val)
            return None

        cam_make = _first_of(["EXIF:Make", "Make", "IFD0:Make", "Maker"])
        cam_model = _first_of(["EXIF:Model", "Model", "IFD0:Model"])
        lens_model = _first_of(["EXIF:LensModel", "EXIF:Lens", "Lens", "Composite:Lens"])
        lens_maker = _first_of(["EXIF:LensMake", "LensMake", "Lens Manufacturer"])

        focal_tag = _first_of(["EXIF:FocalLength", "FocalLength", "Composite:FocalLength"])
        aperture_tag = _first_of(["EXIF:FNumber", "FNumber", "ApertureValue", "Composite:Aperture"])

        focal = _parse_number(focal_tag)
        aperture = _parse_number(aperture_tag)

        # 画像存在と型チェック
        if not hasattr(self, "image_array"):
            print("self.image_array が見つかりません。")
            return False
        img = self.image_array

        # 想定は float32, 0..1 の線形RGB
        if img.dtype != np.float32:
            img = img.astype(np.float32)

        # 画像サイズ
        h, w = img.shape[:2]

        # ---- lensfun DB 検索 ----
        db = lensfunpy.Database()
        cam = None
        lens = None
        try:
            if cam_make and cam_model:
                cams = db.find_cameras(cam_make, cam_model)
                if cams:
                    cam = cams[0]
            if cam is None and cam_model:
                cams = db.find_cameras(None, cam_model)
                if cams:
                    cam = cams[0]
        except Exception:
            cam = None

        try:
            if cam is not None:
                if lens_maker and lens_model:
                    lenses = db.find_lenses(cam, lens_maker, lens_model)
                    if lenses:
                        lens = lenses[0]
                if lens is None and lens_model:
                    lenses = db.find_lenses(cam, None, lens_model)
                    if lenses:
                        lens = lenses[0]
            else:
                # カメラ未特定でも試す（API のバージョンに依存）
                try:
                    if lens_maker and lens_model:
                        lenses = db.find_lenses(None, lens_maker, lens_model)
                        if lenses:
                            lens = lenses[0]
                    elif lens_model:
                        lenses = db.find_lenses(None, None, lens_model)
                        if lenses:
                            lens = lenses[0]
                except Exception:
                    lens = None
        except Exception:
            lens = None

        if lens is None:
            print("lensfun データベースでレンズプロファイルが見つかりませんでした。")
            print("  カメラメーカー:", cam_make, " カメラモデル:", cam_model)
            print("  レンズ情報:", lens_maker, "/", lens_model)
            return False

        # focal/aperture が無ければ lens オブジェクトの推定値を使う
        if focal is None:
            try:
                minf = getattr(lens, "min_focal", None)
                maxf = getattr(lens, "max_focal", None)
                if minf is not None and maxf is not None:
                    focal = float((minf + maxf) / 2.0)
                elif minf is not None:
                    focal = float(minf)
                elif maxf is not None:
                    focal = float(maxf)
            except Exception:
                focal = None
        if aperture is None:
            try:
                max_ap = getattr(lens, "max_aperture", None)
                if max_ap is not None:
                    aperture = float(max_ap)
            except Exception:
                aperture = None

        if focal is None:
            focal = 50.0
        if aperture is None:
            aperture = 5.6

        # ---- Modifier 作成 ----
        try:
            mod = lensfunpy.Modifier(lens, getattr(cam, "crop_factor", 1.0), w, h)
        except Exception as e:
            print("Modifier の作成に失敗:", e)
            return False

        # initialize: 画像は float32 の線形 0..1 にしているので pixel_format=np.float32 を使う
        try:
            mod.initialize(float(focal), float(aperture), distance=1000.0, scale=0.0,
                        pixel_format=np.float32, flags=getattr(lensfunpy, "ModifyFlags", None) and lensfunpy.ModifyFlags.ALL or 0)
        except Exception as e:
            # initialize に失敗しても続行（ただし一部機能は期待通り動かない可能性あり）
            print("Modifier.initialize() の呼び出し時に警告:", e)

        applied_any = False

        # -------- 1) ヴィネット補正（線形イメージ、float32 前提） --------
        try:
            did_vig = mod.apply_color_modification(img)  # in-place で img が更新される（float32で）
            if did_vig:
                print("適用: ヴィネット (vignetting) を補正しました。")
                applied_any = True
            else:
                print("ヴィネット補正データが見つかりませんでした（スキップ）。")
        except Exception as e:
            print("ヴィネット補正中にエラー:", e)

        # -------- 2) サブピクセル / ジオメトリ歪み補正 --------
        try:
            coords_sub = None
            coords = None
            # サブピクセル結合版（チャンネル毎）を試す
            try:
                coords_sub = mod.apply_subpixel_geometry_distortion()
            except Exception:
                coords_sub = None

            if coords_sub is not None:
                # coords_sub の型を float32 にする（OpenCV の remap が float32 を期待するため）
                coords_sub = coords_sub.astype(np.float32, copy=False)
                # まず lensfunpy.util.remap を試す（内部で OpenCV / SciPy を選ぶ）
                try:
                    img_remapped = lensfunpy.util.remap(img, coords_sub)
                    if img_remapped is not None:
                        img = img_remapped
                        print("適用: サブピクセル幾何歪み補正（TCA を含む）を適用しました（util.remap）。")
                        applied_any = True
                    else:
                        # util.remap が None を返すことはほとんどないが一応フォールバック
                        raise RuntimeError("lensfunpy.util.remap が None を返しました。")
                except Exception as e_remap:
                    # util.remap が失敗したら OpenCV を直接使ってチャンネルごとに remap
                    print("util.remap に失敗しました。フォールバックで OpenCV remap を試します。理由:", e_remap)
                    if cv2 is None:
                        print("OpenCV が利用できないためフォールバック remap ができません。")
                    else:
                        # チャンネル毎に remap（coords_sub の形状は (H, W, 3, 2)）
                        try:
                            out = np.zeros_like(img)
                            for c in range(3):
                                map_xy = coords_sub[:, :, c, :]  # (H,W,2)
                                # OpenCV remap は mapX, mapY の2つの single-channel float32 を受けるのが安定
                                map_x = map_xy[:, :, 0].astype(np.float32, copy=False)
                                map_y = map_xy[:, :, 1].astype(np.float32, copy=False)
                                # cv2.remap の引数は src, map1, map2, interpolation, borderMode
                                # src: 単一チャネル。img の各チャネルは float32 0..1
                                src_chan = (img[:, :, c]).astype(np.float32)
                                rem = cv2.remap(src_chan, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                                out[:, :, c] = rem
                            img = out
                            print("適用: サブピクセル幾何歪み補正（TCA を含む）を OpenCV で適用しました。")
                            applied_any = True
                        except Exception as e_cv:
                            print("サブピクセル幾何歪みマップでの remap に失敗しました（OpenCV フォールバック）:", e_cv)
            else:
                # サブピクセルが無ければ通常のジオメトリマップを試す
                try:
                    coords = mod.apply_geometry_distortion()
                except Exception:
                    coords = None
                if coords is not None:
                    coords = coords.astype(np.float32, copy=False)
                    try:
                        img_remapped = lensfunpy.util.remap(img, coords)
                        if img_remapped is not None:
                            img = img_remapped
                            print("適用: 幾何歪み補正（ジオメトリ）を適用しました（util.remap）。")
                            applied_any = True
                        else:
                            raise RuntimeError("lensfunpy.util.remap が None を返しました。")
                    except Exception as e_remap2:
                        print("ジオメトリ remap util.remap に失敗:", e_remap2)
                        if cv2 is None:
                            print("OpenCV が利用できないためフォールバック remap ができません。")
                        else:
                            try:
                                # coords は (H,W,2) -> map_x,map_y
                                map_x = coords[:, :, 0].astype(np.float32, copy=False)
                                map_y = coords[:, :, 1].astype(np.float32, copy=False)
                                out = np.zeros_like(img)
                                for c in range(3):
                                    src_chan = (img[:, :, c]).astype(np.float32)
                                    out[:, :, c] = cv2.remap(src_chan, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                                img = out
                                print("適用: 幾何歪み補正（ジオメトリ）を OpenCV で適用しました。")
                                applied_any = True
                            except Exception as e_cv2:
                                print("ジオメトリマップでの remap に失敗しました（OpenCV フォールバック）:", e_cv2)
                else:
                    print("幾何歪み / TCA の補正データが見つかりません（スキップ）。")
        except Exception as e:
            print("歪み/TCA 補正処理で予期せぬエラー:", e)

        # -------- 3) scale / projection 情報が modifier に設定されているか確認 --------
        try:
            scale_val = getattr(mod, "scale", None)
            if scale_val is not None and scale_val not in (0.0, 1.0):
                print(f"注意: modifier によりスケールが {scale_val} に設定されています（自動スケーリング情報）。")
                applied_any = True
        except Exception:
            pass

        # 最終結果を self.image_array に格納（float32 0..1）
        try:
            # クリップして dtype を float32 にしておく
            img = np.clip(img, 0.0, 1.0).astype(np.float32)
            self.image_array = img
        except Exception as e:
            print("出力画像の変換に失敗しました:", e)
            return False

        if applied_any:
            print("レンズ補正を完了しました。self.image_array を更新しました。")
            return True
        else:
            print("補正を実行しましたが、どの補正も適用されませんでした（プロファイル不足等）。")
            return False



    def _create_tone_lut_from_params(
        self,
        exposure: float = 0.0,
        contrast: int = 0,
        shadow: int = 0,
        highlight: int = 0,
        black: int = 0,
        white: int = 0,
        dtype=np.float32
    ) -> np.ndarray:
        """
        輝度トーン補正用のLUT（Look-Up Table）を作成する関数。
        入力輝度 [0.0, 1.0] に対して、露出補正・トーンカーブ・コントラスト補正を適用した
        出力輝度を返すLUTを構築します。

        Parameters:
            exposure (float): EV補正値（例: +1.0, -0.5など）
            contrast (int): コントラスト補正（-100～+100）
            shadow (int): シャドウ補正（-100～+100）
            highlight (int): ハイライト補正（-100～+100）
            black (int): 黒レベル補正（-100～+100）
            white (int): 白レベル補正（-100～+100）
            

        Returns:
            np.ndarray: shape=(65536,)、値域=[0, 65535]のLUT
        """
        # --- 1. 入力輝度サンプルを作成（65536個） ---
        x_lum = np.linspace(0.0, 1.0, 65536, dtype=np.float32)  # 入力輝度（線形RGBの輝度）

        # --- 2. EV補正 ---
        x_lum_ev = np.clip(x_lum * (2.0 ** exposure), 0.0, 1.0)

        # --- 3. トーンカーブ制御点 ---
        # 代表点の位置
        p5  = 0.05
        p25 = 0.25
        p50 = 0.50
        p75 = 0.75
        p95 = 0.95

        # スライダー値を制御点Yにマッピング
        black_l     = p5  + (p50 - p5)  * (black    / 100.0)
        shadow_l    = p25 + (p50 - p25) * (shadow   / 100.0)
        mid_l        = p50
        highlight_l = p75 + (p95 - p75) * (highlight / 100.0)
        white_l     = p95 + (p95 - p50) * (white    / 100.0)

        xs = np.array([0.0, p5, p25, p50, p75, p95, 1.0], dtype=np.float32)
        ys = np.clip(np.array([0.0, black_l, shadow_l, mid_l, highlight_l, white_l, 1.0], dtype=np.float32), 0.0, 1.0)

        # --- 4. PCHIP補間カーブ ---
        tone_curve = interpolate.PchipInterpolator(xs, ys)

        # --- 5. トーンカーブ適用 ---
        lum_mapped = tone_curve(x_lum_ev)

        # --- 6. コントラスト適用 ---
        c_factor = 1.0 + contrast / 100.0
        lum_contrasted = 0.5 + (lum_mapped - 0.5) * c_factor

        # --- 7. 範囲制限 + uint16に変換 ---
        lut = np.clip(lum_contrasted, 0.0, 1.0) * 65535.0
        return lut.astype(dtype)
    

    def set_tone(self, exposure: float = 0.0, contrast: int = 0, shadow: int = 0, highlight: int = 0, black: int = 0, white: int = 0, mask_name: str = None) -> None: 
        """写真の明るさ関連の調整を設定"""

        if exposure < -10.0 or exposure > 10.0:
            raise ValueError("exposure must be between -10.0 and 10.0")
        if contrast < -100 or contrast > 100:
            raise ValueError("contrast must be between 0 and 100")
        if shadow < -100 or shadow > 100:
            raise ValueError("shadow must be between 0 and 100")
        if highlight < -100 or highlight > 100:
            raise ValueError("highlight must be between 0 and 100")
        if black < -100 or black > 100:
            raise ValueError("black must be between 0 and 100")
        if white < -100 or white > 100:
            raise ValueError("white must be between 0 and 100")
        
        if mask_name is None:
            self.main_adjustments_values["exposure"] = exposure
            self.main_adjustments_values["contrast"] = contrast
            self.main_adjustments_values["shadow"] = shadow
            self.main_adjustments_values["highlight"] = highlight
            self.main_adjustments_values["black"] = black
            self.main_adjustments_values["white"] = white
        else:
            self.mask_adjustments_values[mask_name]["exposure"] = exposure
            self.mask_adjustments_values[mask_name]["contrast"] = contrast
            self.mask_adjustments_values[mask_name]["shadow"] = shadow
            self.mask_adjustments_values[mask_name]["highlight"] = highlight
            self.mask_adjustments_values[mask_name]["black"] = black
            self.mask_adjustments_values[mask_name]["white"] = white
    
    

    
   

    def set_tone_curve(self, control_points_x: list[int] = None, control_points_y: list[int] = None, /, *, curve: np.ndarray = None, mask_name: str = None, curve_graph_save_path: str = None):
        """
        明るさトーンカーブを設定する
        """
        
        if curve is None and control_points_x is None and control_points_y is None:
            raise ValueError("制御点またはトーンカーブを指定してください")
        
        
            
        
        if curve is not None:
            
            
                        
            
            if len(curve) != 65536:
                raise ValueError("リストの長さは65536にしてください")
            
            
            
        if curve is None:
            if control_points_x is None or control_points_y is None:
                raise ValueError("x, yの制御点を指定してください")
            
            if len(control_points_x) != len(control_points_y):
                raise ValueError("制御点の数が一致していません")
            # スプライン補間でトーンカーブを生成
            f = interpolate.PchipInterpolator(control_points_x, control_points_y)
            curve = np.clip(f(np.arange(65536)), 0, 65535).astype(np.uint16)
        

        if mask_name is None:
            self.main_adjustments_values["brightness_tone_curve"] = curve
        else:
            self.mask_adjustments_values[mask_name]["brightness_tone_curve"] = curve


        if curve_graph_save_path:
            plt.figure(figsize=(8, 8))
            plt.plot(list(range(0, 65536)), curve)
            if curve is None and control_points_x is not None:
                for x in control_points_x:
                    plt.plot(x, curve[x], marker='.', color='red')
            plt.grid(True)
            plt.title('brightness_tone_curve')
            plt.savefig(curve_graph_save_path, dpi=300, bbox_inches='tight')
            plt.close()

        
    
    
    

    def set_hls_hue_tone_curve(self, control_points_x: list[int] = None, control_points_y: list[int] = None, /, *, curve: np.ndarray = None, mask_name: str = None, curve_graph_save_path: str = None):
        """
        HLS色相トーンカーブを設定する
        """
        if curve is None and control_points_x is None and control_points_y is None:
            raise ValueError("制御点またはトーンカーブを指定してください")
        
        

        if curve is not None:
            
            
            hue_tone_curve = curve
                    
            
            
        if curve is None:
            if control_points_x is None or control_points_y is None:
                raise ValueError("x, yの制御点を指定してください")
            
            if len(control_points_x) != len(control_points_y):
                raise ValueError("制御点の数が一致していません")
            # スプライン補間で色相調整カーブを生成
            f = interpolate.PchipInterpolator(np.array(control_points_x) * 182, np.array(control_points_y) * 182)
            
            # 0〜65535の範囲で色相調整カーブを生成
            hue_tone_curve = np.clip(f(np.arange(65536)), 0, 65535).astype(np.uint16)
        
        if mask_name is None:
            self.main_adjustments_values["hue_tone_curve"] = hue_tone_curve
        else:
            self.mask_adjustments_values[mask_name]["hue_tone_curve"] = hue_tone_curve


        if curve_graph_save_path:
            plt.figure(figsize=(8, 8))

            # 背景に色相環を表示
            for i in range(360):
                plt.axvline(i, color=mcolors.hsv_to_rgb([i/360, 1, 1]), alpha=0.2)

            # y軸に7本の横線を追加
            y_lines = [0, 60, 120, 180, 240, 300, 360]
            for y in y_lines:
                plt.axhline(y, color=mcolors.hsv_to_rgb([y/360, 1, 1]), alpha=0.5, linewidth=1)
                plt.text(-10, y, f'{y}°', va='center', ha='right')

            x = np.arange(65536) / 182
            y = hue_tone_curve / 182
            plt.plot(x, y, 'k-', linewidth=2)

            if curve is None and control_points_x is not None and control_points_y is not None:
                for x0, y0 in zip(control_points_x, control_points_y):
                    plt.plot([x0, x0], [x0, y0], 'k-', linewidth=1)
                    plt.plot(x0, x0, 'o', color=mcolors.hsv_to_rgb([x0/360, 1, 1]), markersize=10)
                    plt.plot(x0, y0, 'o', color=mcolors.hsv_to_rgb([y0/360, 1, 1]), markersize=10)

            plt.xlabel('original hue')
            plt.ylabel('adjusted hue')
            plt.title('hue_tone_curve')
            plt.xlim(0, 360)
            plt.ylim(0, 360)
            plt.grid(True)
            plt.savefig(curve_graph_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
    

        
    def set_hls_saturation_tone_curve(self, control_points_x: list[int] = None, control_points_y: list[int] = None, /, *, curve: np.ndarray = None, mask_name: np.ndarray = None, curve_graph_save_path: str = None):
        """
        HLS彩度トーンカーブを設定する
        """
        if curve is None and control_points_x is None and control_points_y is None:
            raise ValueError("制御点またはトーンカーブを指定してください")
        
        
            

        

        if curve is None:
            if control_points_x is None or control_points_y is None:
                raise ValueError("x, yの制御点を指定してください")
            
            if len(control_points_x) != len(control_points_y):
                raise ValueError("制御点の数が一致していません")
            
            # スプライン補間で彩度調整カーブを生成
            f = interpolate.PchipInterpolator(np.array(control_points_x) * 182, np.array(control_points_y) / 100 * 65535)
            
            # 0〜65535の範囲で彩度調整カーブを生成
            curve = np.clip(f(np.arange(65536)), 0, 131070).astype(np.float32)
        
        if mask_name is None:
            self.main_adjustments_values["saturation_tone_curve"] = curve
        else:
            self.mask_adjustments_values[mask_name]["saturation_tone_curve"] = curve

        if curve_graph_save_path:
            plt.figure(figsize=(8, 8))

            # 背景に色相環を表示
            for i in range(360):
                color = mcolors.hsv_to_rgb([i/360, 1, 1])
                plt.axvline(i, color=color, alpha=0.2)

            # 彩度調整カーブをプロット
            x = np.arange(65536) / 182
            y = curve / 655.35
            plt.plot(x, y, 'k-', linewidth=2)

            if curve is None and control_points_x is not None and control_points_y is not None:
                for x0, y0 in zip(control_points_x, control_points_y):
                    color = mcolors.hsv_to_rgb([x0/360, 1, 1])
                    plt.plot(x0, y0, 'o', color=color, markersize=10)

            plt.xlabel('hue')
            plt.ylabel('saturation (%)')
            plt.title('saturation_tone_curve')
            plt.xlim(0, 360)
            plt.ylim(0, 200)
            plt.grid(True, alpha=0.3)
            y_ticks = [0, 50, 100, 150, 200]
            plt.yticks(y_ticks)
            plt.axhline(100, color='r', linestyle='--', alpha=0.5)
            plt.text(365, 100, '100%', va='center', ha='left', color='r')

            plt.savefig(curve_graph_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        
    

    def set_hls_lightness_tone_curve(self, control_points_x: list[int] = None, control_points_y: list[int] = None, /, *, curve: np.ndarray = None, mask_name: str = None, curve_graph_save_path: str = None):
        """
        HLS輝度トーンカーブを設定する
        """
        if curve is None and control_points_x is None and control_points_y is None:
            raise ValueError("制御点またはトーンカーブを指定してください")
        
        
            
            
        

        if curve is None:
            if control_points_x is None or control_points_y is None:
                raise ValueError("x, yの制御点を指定してください")
            
            if len(control_points_x) != len(control_points_y):
                raise ValueError("制御点の数が一致していません")
            # スプライン補間で彩度調整カーブを生成
            f = interpolate.PchipInterpolator(np.array(control_points_x) * 182, np.array(control_points_y) / 100 * 65535)
            
            # 0〜65535の範囲で彩度調整カーブを生成
            curve = np.clip(f(np.arange(65536)), 0, 131070).astype(np.float32)
        
        if mask_name is None:
            self.main_adjustments_values["lightness_tone_curve"] = curve
        else:
            self.mask_adjustments_values[mask_name]["lightness_tone_curve"] = curve

        if curve_graph_save_path:
            plt.figure(figsize=(12, 8))

            # 背景に色相環を表示
            for i in range(360):
                color = mcolors.hsv_to_rgb([i/360, 1, 1])
                plt.axvline(i, color=color, alpha=0.2)

            # 輝度調整カーブをプロット
            x = np.arange(65536) / 182
            y = curve / 655.35
            plt.plot(x, y, 'k-', linewidth=2)

            if curve is None and control_points_x is not None and control_points_y is not None:
                for x0, y0 in zip(control_points_x, control_points_y):
                    color = mcolors.hsv_to_rgb([x0/360, 1, 1])
                    plt.plot(x0, y0, 'o', color=color, markersize=10)

            plt.xlabel('hue')
            plt.ylabel('lightness (%)')
            plt.title('lightness_tone_curve')
            plt.xlim(0, 360)
            plt.ylim(0, 200)
            plt.grid(True, alpha=0.3)
            y_ticks = [0, 50, 100, 150, 200]
            plt.yticks(y_ticks)
            plt.axhline(100, color='r', linestyle='--', alpha=0.5)
            plt.text(365, 100, '100%', va='center', ha='left', color='r')

            plt.savefig(curve_graph_save_path, dpi=300, bbox_inches='tight')
            plt.close()

    def set_vignette(self, strength: int, mask_name: str = None):
        """
        周辺減光の強さを設定する関数

        Parameters:
            strength (int): 周辺減光の強さ (-100〜100)
            mask_name (str, optional): マスク名. Defaults to None.
        """
        if strength < -100 or strength > 100:
            raise ValueError("strength must be between -100 and 100")
        
        if mask_name is None:
            self.main_adjustments_values["vignette"] = strength
        else:
            self.mask_adjustments_values[mask_name]["vignette"] = strength
                    
    
    

        

    

    def _create_mask_data_dict(self, mask: np.ndarray, x: int, y: int, inverted: bool) -> dict[str, Any]:

        mask_data_dict = {"ndarray": mask, 
                          "x": x, 
                          "y": y, 
                          "mask_range_value": 0.0, 
                          "wb_temperature": 0, 
                          "wb_tint": 0,
                          "exposure": 0.0,
                          "contrast": 0,
                          "shadow": 0,
                          "highlight": 0,
                          "black": 0,
                          "white": 0,
                          "vignette": 0,

                          "brightness_tone_curve": np.arange(65536),
                          "hue_tone_curve": np.arange(65536),
                          "saturation_tone_curve": np.full((65536,), 65535),
                          "lightness_tone_curve": np.full((65536,), 65535),
                          "inverted": inverted
                          
        }

        return mask_data_dict

        


    def create_ai_mask(self, point: list[int], mask_name: str) -> np.ndarray:
        """
        AIでマスクを作成するメソッド

        sam2.1のAIでマスクを作成するメソッド
        マスクはself.mask_adjustments_values[mask_name]に保存される。
        これにはマスクのndarrayや編集の情報が含まれる。
        Parameters:
            point (list[int]): マスクを作成するための入力ポイント [x, y]

        Returns:
            mask (np.ndarray): 作成されたマスク
            bool型ではありません
            bool_mask = mask > 0 のようにしてbool型に変換してください。
            この閾値を変えるとマスクの範囲を調整できます。
        """
        
        image = self.as_uint8()
        
        with torch.inference_mode():
            predictor.set_image(image)
            
            # 入力ポイントを設定
            input_point = np.array([point])
            input_label = np.array([1])  # 1は前景を示す
            
            # マスクを予測
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
                return_logits=True
            )
        
        # 最初のマスクを取得し、ブール型に変換
        mask = masks[0]
        
        self.mask_adjustments_values[mask_name] = self._create_mask_data_dict(mask, point[0], point[1], False)

        return mask
    
    def delete_mask(self, mask_name: str) -> None:
        """
        マスクを削除するメソッド

        Parameters:
            mask_name (str): 削除するマスクの名前
        """
        del self.mask_adjustments_values[mask_name]
    



    def get_mask_image(self, mask: np.ndarray) -> Image.Image:
        """
        マスク画像のPIL.Imageを取得するメソッド

        Parameters:
            mask (np.ndarray): 表示するマスク

        Returns:
            mask_image (PIL.Image): マスク画像
        """
        mask_image = self.image_array.copy()

        mask_image[mask, 0] = 0.63
        mask_image[mask, 1] *= 0.75
        mask_image[mask, 2] *= 0.75

        return Image.fromarray((mask_image * 255.0).astype(np.uint8))
    


    def show_mask(self, mask: np.ndarray, save_path: str = None):
        """
        マスクを表示するメソッド

        Parameters:
            mask (np.ndarray): 表示するマスク
            save_path (str): 保存する場合のパス Noneの場合は保存しないで表示する
        """

        mask_image = self.get_mask_image(mask)
        if save_path is not None:
            mask_image.save(save_path)
        else:
            mask_image.show()
    

    

        

    

    def get_mask_data_dict(self, mask_name: str) -> dict:
        """
        マスクやマスクの情報を取得するメソッド
        """
        return self.mask_adjustments_values[mask_name]

def estimate_clip_limit_advanced(img: np.ndarray,
                                 min_limit: float = 0.0008,
                                 max_limit: float = 0.005) -> float:
    """
    多指標を使って高精度にclip_limitを推定する。

    特徴量:
      - 局所標準偏差の中央値（分布の中心）
      - skewness（非対称性：暗部が多いか明部が多いか）
      - エントロピー（情報量）

    Args:
        img (np.ndarray): 入力輝度画像（float32, 0〜1）
        min_limit (float): 最小clip_limit
        max_limit (float): 最大clip_limit

    Returns:
        float: 推定されたclip_limit
    """
    # 局所標準偏差（local contrast）
    mean = uniform_filter(img, size=9)
    mean_sq = uniform_filter(img**2, size=9)
    local_var = mean_sq - mean**2
    local_std = np.sqrt(np.clip(local_var, 0, None))

    std_median = np.median(local_std)
    std_score = 1.0 - np.clip(std_median / 0.1, 0.0, 1.0)  # 大きいほど補正を弱く

    # skewness（輝度の偏り）
    flat = img.flatten()
    sk = np.clip(np.abs(skew(flat)), 0.0, 1.5)
    skew_score = 1.0 - sk / 1.5  # 非対称 → clip強め

    # エントロピー（輝度ヒストグラムの情報量）
    hist, _ = np.histogram(flat, bins=256, range=(0.0, 1.0), density=True)
    hist += 1e-6  # log(0)防止
    ent = entropy(hist)
    entropy_score = 1.0 - np.clip(ent / 5.5, 0.0, 1.0)  # 情報量少 → clip強め

    # スコアの重み付き平均（必要なら調整可能）
    score = 0.5 * std_score + 0.3 * entropy_score + 0.2 * skew_score

    # 線形補間でclip_limit決定
    clip = min_limit + (max_limit - min_limit) * score
    return clip


def estimate_clip_limit(img: np.ndarray, min_limit=0.001, max_limit=0.005) -> float:
    """
    入力画像から自動で適したclip_limitを推定する。
    局所コントラストが低いほどclip_limitを大きく設定。
    
    Args:
        img (np.ndarray): 入力輝度画像 (float32, [0.0, 1.0])
        min_limit (float): 最小clip_limit（例: 0.005）
        max_limit (float): 最大clip_limit（例: 0.05）

    Returns:
        float: 推定されたclip_limit
    """
    # 輝度の局所分散を調べる（local contrast）
    mean = uniform_filter(img, size=9)
    mean_sq = uniform_filter(img**2, size=9)
    local_var = mean_sq - mean**2
    local_std = np.sqrt(np.clip(local_var, 0, None))

    # 平均局所コントラスト
    avg_local_contrast = np.mean(local_std)

    # 正規化（0.0〜1.0）してclip_limitを線形に補間
    # 0.0のとき最大clip_limit、1.0のとき最小clip_limit
    norm_contrast = np.clip(avg_local_contrast / 0.1, 0.0, 1.0)
    estimated = max_limit - (max_limit - min_limit) * norm_contrast
    return estimated



def apply_gamma(img: np.ndarray, gamma=(2.222, 4.5/255.0)) -> np.ndarray:
    """
    float32画像に rawpy の gamma=(g, c) と同じ補正を適用する関数

    Parameters
    ----------
    img : np.ndarray
        入力画像 (float32, 値域 [0, 1])
    gamma : tuple (g, c)
        gamma[0] = ガンマ値 (例: 2.222)
        gamma[1] = スロープ (例: 4.5/255 ≈ 0.018)

    Returns
    -------
    np.ndarray
        ガンマ補正後の画像 (float32, 値域 [0, 1])
    """
    g, c = gamma
    c /= 255.0
    img = np.clip(img, 0.0, 1.0)

    # スロープからしきい値を計算
    threshold = (c / (g - 1.0)) ** g

    out = np.where(
        img < threshold,
        img * (c / (g - 1.0)),
        (1.0 + c) * np.power(img, 1.0 / g) - c
    )
    return out.astype(np.float32)


class RAWImageEditor(RAWImageEditorBase):
    # Class variables for wgpu resources
    device: wgpu.GPUDevice = None
    clahe_pipeline: wgpu.GPUComputePipeline = None
    clip_0_1_pipeline: wgpu.GPUComputePipeline = None
    tone_curve_lut_pipeline: wgpu.GPUComputePipeline = None
    rgb_to_hls_pipeline: wgpu.GPUComputePipeline = None
    hls_to_rgb_pipeline: wgpu.GPUComputePipeline = None
    to_linear_pipeline: wgpu.GPUComputePipeline = None
    to_srgb_pipeline: wgpu.GPUComputePipeline = None
    tone_curve_by_hue_pipeline: wgpu.GPUComputePipeline = None
    white_balance_pipeline: wgpu.GPUComputePipeline = None
    vignette_effect_pipeline: wgpu.GPUComputePipeline = None
    
    def __init__(self, file_path: str | None = None, 
                 image_array: np.ndarray | None = None, 
                 apply_clahe: bool = False, 
                 lens_correction: bool = False,
                 gamma: tuple[float, float] = (1.0, 1.0), 
                 metadata: photo_metadata.Metadata | None = None, 
                 exiftool_path: str | None = None, 
                 jp_tags_json_path: str | None = None,
                 is_raw_ignore=False):
        
        """

        RAWImageEditorのwgpuバックエンド


        Parameters:
            file_path (str): RAWファイルのパス
            image_array (numpy.ndarray): 写真のndarray
            metadata (photo_metadata.Metadata): 写真のメタデータ
            
            exiftool_path (str): exiftoolのpath
            jp_tags_json_path (str): exiftool_Japanese_tag.jsonのpath
        
        メタデータのみを指定することはできません。
        image_arrayを指定した場合はfile_pathは指定できません。
        image_arrayを指定する場合は8bit (0~255)ではなく0~1 (float32)に正規化してください。

        RAW現像を行うclass。 RAW写真の読み込み、postprocess (rawpy)、編集、エクスポートができる。
        RAW写真を読み込み、postprocessで最初の処理を行い、ndarrayにする。
        
        インスタンス変数:
            file_path (str): RAWファイルのパス
            image_array (numpy.ndarray): 写真のndarray
            
            metadata (photo_metadata.Metadata): 写真のメタデータ
        """
        
        super().__init__(file_path=file_path, 
                         image_array=image_array, 
                         metadata=metadata, 
                         exiftool_path=exiftool_path, 
                         jp_tags_json_path=jp_tags_json_path)
        
        is_raw = True if is_raw_ignore else self.is_raw

        if lens_correction and is_raw:
            self.lens_correction()
        if gamma != (1.0, 1.0) and is_raw:
            self.image_array = apply_gamma(self.image_array, gamma=gamma);
           
        
        if apply_clahe and is_raw:
            clip = 0.002
            print(f"clip: {clip}")
            # self.image_array = self.clahe.clahe_rgb(
            #                              self.image_array, 
            #                              tile_grid_y=8,
            #                              tile_grid_x=8, 
            #                              clip_limit=clip,
            #                              num_bins=512
            #                              )
            
        self.initial_image_array = self.image_array.copy()
        
        

        self.full_mask = self.device.create_buffer_with_data(data=np.ones((self.height * self.width,), dtype=np.uint32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
        self.hue_channels = self.device.create_buffer_with_data(data=np.array([0], dtype=np.int32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
        self.full_channels = self.device.create_buffer_with_data(data=np.array([0, 1, 2], dtype=np.int32).tobytes(), usage=wgpu.BufferUsage.STORAGE)

        self.num_pixels: int = self.height * self.width
        self.num_elements: int = self.num_pixels * 3
        



    def copy(self):
        return copy.copy(self)


    def apply_adjustments(self):
        """すべての編集を適用する関数"""
        
        # numpy -> wgpu buffer
        self.image_array = np.clip(self.image_array, 0.0, 1.0)
        image_buffer = self.device.create_buffer_with_data(data=self.image_array.flatten().astype(np.float32).tobytes(), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        
        # 1. sRGB -> Linear (in-place)
        self.to_linear(image_buffer)

        # 2. --- LinearRGB 空間での調整 ---
        # 2.1. メイン調整
        self._adjustment_tone(
            image_buffer,
            exposure=self.main_adjustments_values["exposure"],
            contrast=self.main_adjustments_values["contrast"],
            shadow=self.main_adjustments_values["shadow"],
            highlight=self.main_adjustments_values["highlight"],
            black=self.main_adjustments_values["black"],
            white=self.main_adjustments_values["white"],
            mask=None
        )
        self._adjustment_tone_curve(
            image_buffer,
            self.main_adjustments_values["brightness_tone_curve"],
            mask=None
        )
        self._adjustment_whitebalance(
            image_buffer,
            temperature=self.main_adjustments_values["wb_temperature"],
            tint=self.main_adjustments_values["wb_tint"],
            mask=None
        )
        self._adjustment_vignette(
            image_buffer,
            self.main_adjustments_values["vignette"],
            mask=None
        )

        # 2.2. マスクごとの調整
        for mask_data in self.mask_adjustments_values.values():
            bool_mask = mask_data["ndarray"] > mask_data["mask_range_value"]
            self._adjustment_tone(
                image_buffer,
                exposure=mask_data["exposure"],
                contrast=mask_data["contrast"],
                shadow=mask_data["shadow"],
                highlight=mask_data["highlight"],
                black=mask_data["black"],
                white=mask_data["white"],
                mask=bool_mask
            )
            self._adjustment_tone_curve(
                image_buffer,
                mask_data["brightness_tone_curve"],
                mask=bool_mask
            )
            self._adjustment_whitebalance(
                image_buffer,
                temperature=mask_data["wb_temperature"],
                tint=mask_data["wb_tint"],
                mask=bool_mask
            )
            self._adjustment_vignette(
                image_buffer,
                mask_data["vignette"],
                mask=bool_mask
            )

        # 3. Linear -> sRGB (in-place)
        self.to_srgb(image_buffer)
        
        # 4. sRGB -> HLS (in-place)
        self.rgb_to_hls_image(image_buffer)

        # 5. --- HLS 空間での調整 ---
        # 5.1. メイン調整
        self._adjustment_hls_hue_tone_curve(
            image_buffer,
            self.main_adjustments_values["hue_tone_curve"],
            mask=None
        )
        self._adjustment_hls_saturation_tone_curve(
            image_buffer,
            self.main_adjustments_values["saturation_tone_curve"],
            mask=None
        )
        self._adjustment_hls_lightness_tone_curve(
            image_buffer,
            self.main_adjustments_values["lightness_tone_curve"],
            mask=None
        )

        # 5.2. マスクごとの調整
        for mask_data in self.mask_adjustments_values.values():
            bool_mask = mask_data["ndarray"] > mask_data["mask_range_value"]
            self._adjustment_hls_hue_tone_curve(
                image_buffer,
                mask_data["hue_tone_curve"],
                mask=bool_mask
            )
            self._adjustment_hls_saturation_tone_curve(
                image_buffer,
                mask_data["saturation_tone_curve"],
                mask=bool_mask
            )
            self._adjustment_hls_lightness_tone_curve(
                image_buffer,
                mask_data["lightness_tone_curve"],
                mask=bool_mask
            )

        # 6. HLS -> sRGB (in-place)
        self.hls_to_rgb_image(image_buffer)
        
        # 7. 最終的なクリップ
        self._clip_0_1(image_buffer)
        
        # wgpu buffer -> numpy
        mapped_data = self.device.queue.read_buffer(image_buffer, 0, image_buffer.size)
        self.image_array = np.frombuffer(mapped_data, dtype=np.float32).reshape(self.height, self.width, 3)
        


    def _calculate_workgroups_and_stride(self, num_items: int) -> tuple[tuple[int, int, int], int]:
        limit = 65535
        workgroup_size = 256
        
        groups_total = (num_items + workgroup_size - 1) // workgroup_size
        
        if groups_total > limit:
            groups_y = (groups_total + limit - 1) // limit
            groups_x = limit
        else:
            groups_y = 1
            groups_x = groups_total
            
        workgroups = (groups_x, groups_y, 1)
        stride_y = groups_x * workgroup_size
        
        return workgroups, stride_y

    def _run_compute_shader(self, pipeline, buffer_bindings: dict[int, wgpu.GPUBuffer], params_binding: int, params_data, workgroups):
        command_encoder = self.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        
        params_buffer = self.device.create_buffer_with_data(data=params_data, usage=wgpu.BufferUsage.UNIFORM)
        
        bindings = []
        for binding_idx, buffer in buffer_bindings.items():
            bindings.append({"binding": binding_idx, "resource": {"buffer": buffer, "offset": 0, "size": buffer.size}})
        
        bindings.append({"binding": params_binding, "resource": {"buffer": params_buffer, "offset": 0, "size": params_buffer.size}})

        bind_group = self.device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=bindings)
        
        compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
        compute_pass.dispatch_workgroups(*workgroups)
        compute_pass.end()
        self.device.queue.submit([command_encoder.finish()])

    def _clip_0_1(self, img_buffer: wgpu.GPUBuffer):
        """wgpuで0-1にクリップする"""
        workgroups, stride_y = self._calculate_workgroups_and_stride(self.num_elements)
        params_data = np.array([self.num_elements, stride_y], dtype=np.uint32).tobytes()
        self._run_compute_shader(self.clip_0_1_pipeline, {0: img_buffer}, 1, params_data, workgroups)

    def _adjustment_whitebalance(self, rgb_image_buffer: wgpu.GPUBuffer, temperature: int, tint: int, mask: np.ndarray = None):
        r_gain = np.float32(1.0 + 0.5 * (temperature / 100))
        b_gain = np.float32(1.0 - 0.5 * (temperature / 100))
        g_gain = np.float32(1.0 - 0.25 * (tint / 100))
        
        if mask is not None:
            mask_buffer = self.device.create_buffer_with_data(data=mask.flatten().astype(np.uint32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
        else:
            mask_buffer = self.full_mask
        
        workgroups, stride_y = self._calculate_workgroups_and_stride(self.num_pixels)
        
        params_dtype = np.dtype([
            ('num_pixels', np.uint32),
            ('r_gain', np.float32),
            ('g_gain', np.float32),
            ('b_gain', np.float32),
            ('stride_y', np.uint32)
        ])
        params_array = np.array([(self.num_pixels, r_gain, g_gain, b_gain, stride_y)], dtype=params_dtype)
        
        self._run_compute_shader(self.white_balance_pipeline, {0: rgb_image_buffer, 3: mask_buffer}, 7, params_array.tobytes(), workgroups)

    def rgb_to_hls_image(self, rgb_image_buffer: wgpu.GPUBuffer):
        """
        RGB画像をHLS画像に変換する関数
        """
        workgroups, stride_y = self._calculate_workgroups_and_stride(self.num_pixels)
        params_data = np.array([self.num_elements, stride_y], dtype=np.uint32).tobytes()
        self._run_compute_shader(self.rgb_to_hls_pipeline, {0: rgb_image_buffer}, 1, params_data, workgroups)

    def hls_to_rgb_image(self, hls_image_buffer: wgpu.GPUBuffer):
        """
        HLS画像をRGB画像に変換する関数
        """
        workgroups, stride_y = self._calculate_workgroups_and_stride(self.num_pixels)
        params_data = np.array([self.num_elements, stride_y], dtype=np.uint32).tobytes()
        self._run_compute_shader(self.hls_to_rgb_pipeline, {0: hls_image_buffer}, 1, params_data, workgroups)

    def to_linear(self, img_buffer: wgpu.GPUBuffer):
        """sRGB to linear"""
        workgroups, stride_y = self._calculate_workgroups_and_stride(self.num_elements)
        params_data = np.array([self.num_elements, stride_y], dtype=np.uint32).tobytes()
        self._run_compute_shader(self.to_linear_pipeline, {0: img_buffer}, 1, params_data, workgroups)

    def to_srgb(self, img_buffer: wgpu.GPUBuffer):
        """linear to sRGB"""
        workgroups, stride_y = self._calculate_workgroups_and_stride(self.num_elements)
        params_data = np.array([self.num_elements, stride_y], dtype=np.uint32).tobytes()
        self._run_compute_shader(self.to_srgb_pipeline, {0: img_buffer}, 1, params_data, workgroups)

    def _adjustment_tone(
        self,
        img_buffer: wgpu.GPUBuffer,
        exposure: float = 0.0,
        contrast: int = 0,
        shadow: int = 0,
        highlight: int = 0,
        black: int = 0,
        white: int = 0,
        mask: np.ndarray = None,
    ):
        lut_np = self._create_tone_lut_from_params(exposure=exposure, 
                                                   contrast=contrast, 
                                                   shadow=shadow, 
                                                   highlight=highlight, 
                                                   black=black, 
                                                   white=white,
                                                   dtype=np.float32)
        lut_buffer = self.device.create_buffer_with_data(data=lut_np.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        
        if mask is not None:
            mask_buffer = self.device.create_buffer_with_data(data=mask.flatten().astype(np.uint32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
        else:
            mask_buffer = self.full_mask
        
        workgroups, stride_y = self._calculate_workgroups_and_stride(self.num_pixels)
        params_data = np.array([3, self.num_pixels, stride_y], dtype=np.uint32).tobytes()
        self._run_compute_shader(self.tone_curve_lut_pipeline, {0: img_buffer, 2: lut_buffer, 3: mask_buffer, 4: self.full_channels}, 5, params_data, workgroups)

    def _adjustment_tone_curve(self, linear_rgb_image_buffer: wgpu.GPUBuffer, curve: np.ndarray, mask: np.ndarray = None):
        if mask is not None:
            mask_buffer = self.device.create_buffer_with_data(data=mask.flatten().astype(np.uint32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
        else:
            mask_buffer = self.full_mask
        
        curve_buffer = self.device.create_buffer_with_data(data=curve.astype(np.float32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
        
        workgroups, stride_y = self._calculate_workgroups_and_stride(self.num_pixels)
        params_data = np.array([3, self.num_pixels, stride_y], dtype=np.uint32).tobytes()
        self._run_compute_shader(self.tone_curve_lut_pipeline, {0: linear_rgb_image_buffer, 2: curve_buffer, 3: mask_buffer, 4: self.full_channels}, 5, params_data, workgroups)

    def _adjustment_hls_hue_tone_curve(self, hls_image_buffer: wgpu.GPUBuffer, curve: np.ndarray, mask: np.ndarray = None):
        if mask is not None:
            mask_buffer = self.device.create_buffer_with_data(data=mask.flatten().astype(np.uint32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
        else:
            mask_buffer = self.full_mask
        
        curve_buffer = self.device.create_buffer_with_data(data=curve.astype(np.float32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
        
        workgroups, stride_y = self._calculate_workgroups_and_stride(self.num_pixels)
        params_data = np.array([1, self.num_pixels, stride_y], dtype=np.uint32).tobytes()
        self._run_compute_shader(self.tone_curve_lut_pipeline, {0: hls_image_buffer, 2: curve_buffer, 3: mask_buffer, 4: self.hue_channels}, 5, params_data, workgroups)

    def _adjustment_hls_saturation_tone_curve(self, hls_image_buffer: wgpu.GPUBuffer, curve: np.ndarray, mask: np.ndarray = None):
        if mask is not None:
            mask_buffer = self.device.create_buffer_with_data(data=mask.flatten().astype(np.uint32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
        else:
            mask_buffer = self.full_mask
        
        curve_buffer = self.device.create_buffer_with_data(data=curve.astype(np.float32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
        
        workgroups, stride_y = self._calculate_workgroups_and_stride(self.num_pixels)
        params_data = np.array([0, 2, self.num_pixels, stride_y], dtype=np.uint32).tobytes()
        self._run_compute_shader(self.tone_curve_by_hue_pipeline, {0: hls_image_buffer, 2: curve_buffer, 3: mask_buffer}, 6, params_data, workgroups)

    def _adjustment_hls_lightness_tone_curve(self, hls_image_buffer: wgpu.GPUBuffer, curve: np.ndarray, mask: np.ndarray = None):
        if mask is not None:
            mask_buffer = self.device.create_buffer_with_data(data=mask.flatten().astype(np.uint32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
        else:
            mask_buffer = self.full_mask
        
        curve_buffer = self.device.create_buffer_with_data(data=curve.astype(np.float32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
        
        workgroups, stride_y = self._calculate_workgroups_and_stride(self.num_pixels)
        params_data = np.array([0, 1, self.num_pixels, stride_y], dtype=np.uint32).tobytes()
        self._run_compute_shader(self.tone_curve_by_hue_pipeline, {0: hls_image_buffer, 2: curve_buffer, 3: mask_buffer}, 6, params_data, workgroups)

    def _adjustment_vignette(self, rgb_image_buffer: wgpu.GPUBuffer, strength: int, mask: np.ndarray = None):
        """wgpuで周辺減光を適用する"""
        if strength == 0:
            return

        if mask is not None:
            mask_buffer = self.device.create_buffer_with_data(data=mask.flatten().astype(np.uint32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
        else:
            mask_buffer = self.full_mask
        
        strength_float = np.float32((-strength / 100.0) * 2.0)

        workgroups, stride_y = self._calculate_workgroups_and_stride(self.num_pixels)
        
        params_dtype = np.dtype([
            ('width', np.uint32),
            ('height', np.uint32),
            ('strength', np.float32),
            ('num_pixels', np.uint32),
            ('stride_y', np.uint32)
        ])
        params_array = np.array([(self.width, self.height, strength_float, self.num_pixels, stride_y)], dtype=params_dtype)

        self._run_compute_shader(self.vignette_effect_pipeline, {0: rgb_image_buffer, 3: mask_buffer}, 8, params_array.tobytes(), workgroups)
