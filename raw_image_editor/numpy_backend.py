# raw_image_editor.numpy_backend.py

import numpy as np
import photo_metadata
from typing import Tuple, Dict, Optional, Union, Any
import copy
from numba import njit, prange
from scipy.ndimage import uniform_filter
from scipy.stats import skew, entropy
from .base import RAWImageEditorBase





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


@njit
def compute_cdf(tile: np.ndarray, clip_limit: float, num_bins: int) -> np.ndarray:
    """
    タイルごとにヒストグラムを計算し、クリップ制限付きCDF（LUT）を生成します。
    Args:
        tile (np.ndarray): 入力タイル（float32, [0.0, 1.0]）
        clip_limit (float): ヒストグラムクリップ制限（0.0-1.0の比率）
        num_bins (int): ヒストグラムビン数
    Returns:
        np.ndarray: 正規化されたCDF, shape=(num_bins,), float32
    """
    # タイルサイズ取得
    h, w = tile.shape
    # ビン配列初期化
    hist = np.zeros(num_bins, dtype=np.float32)
    # ヒストグラム計算
    for y in range(h):
        for x in range(w):
            idx = int(tile[y, x] * (num_bins - 1))
            if idx < 0:
                idx = 0
            elif idx >= num_bins:
                idx = num_bins - 1
            hist[idx] += 1.0
    # クリップ＆再分配
    max_per_bin = clip_limit * h * w
    excess = 0.0
    for i in range(num_bins):
        if hist[i] > max_per_bin:
            excess += hist[i] - max_per_bin
            hist[i] = max_per_bin
    redistribute = excess / num_bins
    for i in range(num_bins):
        hist[i] += redistribute
    # CDF計算
    cdf = np.empty(num_bins, dtype=np.float32)
    cdf[0] = hist[0]
    for i in range(1, num_bins):
        cdf[i] = cdf[i - 1] + hist[i]
    # 正規化
    total = cdf[-1]
    for i in range(num_bins):
        cdf[i] /= total
    return cdf

@njit(parallel=True)
def clahe_lut(img: np.ndarray,
              tile_grid_y: int = 8,
              tile_grid_x: int = 8,
              clip_limit: float = 0.01,
              num_bins: int = 256) -> np.ndarray:
    """
    NumPy+Numbaで実装したLUTベースCLAHE（輝度用）
    Args:
        img (np.ndarray): 入力輝度画像（float32, [0.0, 1.0]）, shape=(H, W)
        tile_grid_y (int): タイルの縦分割数
        tile_grid_x (int): タイルの横分割数
        clip_limit (float): クリップ制限率
        num_bins (int): ヒストグラムビン数
    Returns:
        np.ndarray: CLAHE適用後の輝度画像（float32, [0.0, 1.0]）, shape=(H, W)
    """
    H, W = img.shape
    tile_h = H // tile_grid_y
    tile_w = W // tile_grid_x
    # 各タイルのCDFを保存する配列
    maps = np.zeros((tile_grid_y, tile_grid_x, num_bins), dtype=np.float32)
    # タイルごとにCDF計算
    for i in prange(tile_grid_y):
        for j in range(tile_grid_x):
            y0 = i * tile_h
            y1 = y0 + tile_h if i < tile_grid_y - 1 else H
            x0 = j * tile_w
            x1 = x0 + tile_w if j < tile_grid_x - 1 else W
            tile = img[y0:y1, x0:x1]
            maps[i, j] = compute_cdf(tile, clip_limit, num_bins)
    # 出力画像
    out = np.empty_like(img)
    # 各ピクセルでCDFを補間
    for y in prange(H):
        for x in range(W):
            # ビンインデックス
            val = img[y, x]
            bin_idx = int(val * (num_bins - 1))
            if bin_idx < 0:
                bin_idx = 0
            elif bin_idx >= num_bins:
                bin_idx = num_bins - 1
            # タイル内位置
            fy = y / tile_h - 0.5
            fx = x / tile_w - 0.5
            i = int(np.floor(fy))
            j = int(np.floor(fx))
            dy = fy - i
            dx = fx - j
            # 周囲4タイルからCDF値取得
            def get(ii, jj):
                if ii < 0:
                    ii = 0
                elif ii >= tile_grid_y:
                    ii = tile_grid_y - 1
                if jj < 0:
                    jj = 0
                elif jj >= tile_grid_x:
                    jj = tile_grid_x - 1
                return maps[ii, jj, bin_idx]
            v00 = get(i, j)
            v01 = get(i, j + 1)
            v10 = get(i + 1, j)
            v11 = get(i + 1, j + 1)
            # 双線形補間
            out[y, x] = (1 - dy) * (1 - dx) * v00 + \
                         (1 - dy) * dx * v01 + \
                         dy * (1 - dx) * v10 + \
                         dy * dx * v11
    return out


def clahe_rgb(rgb_img: np.ndarray,
              tile_grid_y: int = 8,
              tile_grid_x: int = 8,
              clip_limit: float = 0.003,
              num_bins: int = 512) -> np.ndarray:
    """
    RGB画像に対して、輝度ベースのCLAHEを適用し色を保持します。
    Args:
        rgb_img (np.ndarray): 入力RGB画像（float32, [0.0, 1.0]）, shape=(H, W, 3)
        tile_grid_y (int): タイルの縦分割数
        tile_grid_x (int): タイルの横分割数
        clip_limit (float): クリップ制限率
        num_bins (int): ヒストグラムビン数
    Returns:
        np.ndarray: 補正後RGB画像（float32, [0.0, 1.0]）, shape=(H, W, 3)
    """
    # 輝度計算（BT.709）
    lum = 0.2126 * rgb_img[..., 0] + 0.7152 * rgb_img[..., 1] + 0.0722 * rgb_img[..., 2]  
    # CLAHE適用
    lum_eq = clahe_lut(lum.astype(np.float32), tile_grid_y, tile_grid_x, clip_limit, num_bins)
    # 比率適用
    ratio = lum_eq / (lum + 1e-6)
    out = rgb_img * ratio[..., np.newaxis]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


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


class RAWImageEditorNumpy(RAWImageEditorBase):
    
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

        RAWImageEditorのNumPyバックエンド


        Parameters:
            file_path (str): RAWファイルのパス
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
            file_path (str): RAWファイルのパス
            image_array (numpy.ndarray): 写真のndarray
            
            metadata (photo_metadata.Metadata): 写真のメタデータ
        """
        
        super().__init__(file_path=file_path, 
                         image_array=image_array, 
                         metadata=metadata, 
                         exiftool_path=exiftool_path, 
                         jp_tags_json_path=jp_tags_json_path)
        
        
        is_raw = self.is_raw if self.is_raw is not None else False
        is_raw = True if is_raw_ignore else is_raw

        if lens_correction and is_raw:
            self.lens_correction()
        if gamma != (1.0, 1.0) and is_raw:
            self.image_array = apply_gamma(self.image_array, gamma=gamma)
            
        
        if apply_clahe and is_raw:
            clip = 0.002
            print(f"clip: {clip}")
            self.image_array = clahe_rgb(
                                         self.image_array, 
                                         tile_grid_y=8,
                                         tile_grid_x=8, 
                                         clip_limit=clip,
                                         num_bins=512
                                         )
            
        self.initial_image_array = self.image_array.copy()
            
            


    def copy(self):
        return copy.copy(self)
    

        

    def apply_adjustments(self):
        """すべての編集を適用する関数"""
        
        self.image_array = np.clip(self.image_array, 0.0, 1.0)
        linear_rgb_image = self.to_linear(self.image_array)

        # 1. RGB 空間でクリップし、メインのホワイトバランス調整を行う
        
        linear_rgb_image = self._adjustment_tone(
            linear_rgb_image,
            exposure=self.main_adjustments_values["exposure"],
            contrast=self.main_adjustments_values["contrast"],
            shadow=self.main_adjustments_values["shadow"],
            highlight=self.main_adjustments_values["highlight"],
            black=self.main_adjustments_values["black"],
            white=self.main_adjustments_values["white"],
            mask=None
        )

        linear_rgb_image = np.clip(linear_rgb_image, 0.0, 1.0)


        # 3. メインのトーンカーブと HLS 調整を適用
        linear_rgb_image = self._adjustment_tone_curve(
            linear_rgb_image,
            self.main_adjustments_values["brightness_tone_curve"]
        )
        self.image_array = self.to_srgb(linear_rgb_image)
        self.image_array = self._adjustment_whitebalance(
            self.image_array,
            temperature=self.main_adjustments_values["wb_temperature"],
            tint=self.main_adjustments_values["wb_tint"]
        )

        
        
        # 2. HLS へ変換し、クリップする（ここで一度だけ変換）
        self.hls_image_array = self.rgb_to_hls_image(self.image_array)
        self.hls_image_array = np.clip(self.hls_image_array, 0.0, 1.0)

        
        self.hls_image_array = self._adjustment_hls_hue_tone_curve(
            self.hls_image_array,
            self.main_adjustments_values["hue_tone_curve"]
        )
        self.hls_image_array = self._adjustment_hls_saturation_tone_curve(
            self.hls_image_array,
            self.main_adjustments_values["saturation_tone_curve"]
        )
        self.hls_image_array = self._adjustment_hls_lightness_tone_curve(
            self.hls_image_array,
            self.main_adjustments_values["lightness_tone_curve"]
        )

        # 4. 各マスクに対する HLS 空間でのトーンカーブ調整を適用
        for mask_data in self.mask_adjustments_values.values():
            bool_mask = mask_data["ndarray"] > mask_data["mask_range_value"]
            
            

            self.hls_image_array = self._adjustment_tone_curve(
                self.hls_image_array,
                mask_data["brightness_tone_curve"],
                mask=bool_mask
            )
            self.hls_image_array = self._adjustment_hls_hue_tone_curve(
                self.hls_image_array,
                mask_data["hue_tone_curve"],
                mask=bool_mask
            )
            self.hls_image_array = self._adjustment_hls_saturation_tone_curve(
                self.hls_image_array,
                mask_data["saturation_tone_curve"],
                mask=bool_mask
            )
            self.hls_image_array = self._adjustment_hls_lightness_tone_curve(
                self.hls_image_array,
                mask_data["lightness_tone_curve"],
                mask=bool_mask
            )

        # 5. 最終的な HLS を RGB に変換（ここで一度だけ）
        self.image_array = self.hls_to_rgb_image(self.hls_image_array)

        # 6. マスクごとのホワイトバランス調整（RGB 空間で）
        for mask_data in self.mask_adjustments_values.values():
            bool_mask = mask_data["ndarray"] > mask_data["mask_range_value"]
            self.image_array = self._adjustment_tone(
                self.image_array,
                exposure=mask_data["exposure"],
                contrast=mask_data["contrast"],
                shadow=mask_data["shadow"],
                highlight=mask_data["highlight"],
                black=mask_data["black"],
                white=mask_data["white"],
                mask=bool_mask)
            self.image_array = self._adjustment_whitebalance(
                self.image_array,
                temperature=mask_data["wb_temperature"],
                tint=mask_data["wb_tint"],
                mask=bool_mask
            )

        # 7. 最終的なクリップと HLS 更新（両方の状態を一致させる）
        self.image_array = np.clip(self.image_array, 0.0, 1.0)
        self.hls_image_array = np.clip(self.hls_image_array, 0.0, 1.0)
        self.hls_image_array = self.rgb_to_hls_image(self.image_array)


    
        
        
    
    
    
    
    
    
        
        


    
    
    


    
    def _adjustment_whitebalance(self, rgb_image: np.ndarray, temperature: int, tint: int, mask: np.ndarray = None):
        
        # チャンネル毎のゲイン計算
        r_gain = 1.0 + 0.5 * (temperature / 100)  # 温度が高いほど赤を増加
        b_gain = 1.0 - 0.5 * (temperature / 100)  # 温度が低いほど青を増加
        g_gain = 1.0 - 0.25 * (tint / 100)  # ティントが低いほど緑を増加
        

        

        if mask is not None:
            rgb_image[mask, 0] *= r_gain  # R
            rgb_image[mask, 1] *= g_gain  # G
            rgb_image[mask, 2] *= b_gain  # B
        else: 
            rgb_image[:, :, 0] *= r_gain  # R
            rgb_image[:, :, 1] *= g_gain  # G
            rgb_image[:, :, 2] *= b_gain  # B
        
        return np.clip(rgb_image, 0.0, 1.0)
        
        

    
   
        
        
        

    
        
        

    def rgb_to_hls_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        RGB画像をHLS画像に変換する関数
        """
        # 入力が3次元配列（高さ、幅、チャンネル）であることを確認
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError("入力は3チャンネルのRGB画像である必要があります")
        # 0-65535の範囲に正規化
        rgb_normalized = rgb_image.copy()
        
        r, g, b = rgb_normalized[:,:,0], rgb_normalized[:,:,1], rgb_normalized[:,:,2]
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        
        l = (max_c + min_c) / 2
        s = np.zeros_like(l)
        h = np.zeros_like(l)
        
        delta = max_c - min_c
        
        # 彩度の計算
        mask = (max_c != min_c)
        s[mask] = np.where(l[mask] > 0.5, 
                        delta[mask] / (2 - max_c[mask] - min_c[mask]),
                        delta[mask] / (max_c[mask] + min_c[mask]))
        
        # 色相の計算
        mask_r = (max_c == r) & mask
        h[mask_r] = (g[mask_r] - b[mask_r]) / delta[mask_r] + np.where(g[mask_r] < b[mask_r], 6, 0)
        
        mask_g = (max_c == g) & mask
        h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
        
        mask_b = (max_c == b) & mask
        h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
        
        h /= 6
        
        
        
        
        return np.dstack((h, l, s)).astype(np.float32)

    def hls_to_rgb_image(self, hls_image: np.ndarray) -> np.ndarray:
        """
        HLS画像をRGB画像に変換する関数
        """
        if hls_image.ndim != 3 or hls_image.shape[2] != 3:
            raise ValueError("入力は3チャンネルのHLS画像である必要があります")

        # チャンネルに分割
        h = hls_image[:, :, 0]
        l = hls_image[:, :, 1]
        s = hls_image[:, :, 2]

        # q, pの計算
        q = np.where(l < 0.5, l * (1 + s), l + s - l * s)
        p = 2 * l - q

        # 各チャンネル用の補助t値を計算
        def _hue2rgb(p, q, t):
            """
            HLS→RGB変換で使う補助関数。
            p, q, tはすべて同じ形状でndarray。
            """
            t = t % 1.0  # tが1を超えたり負になったりしたとき用
            res = np.empty_like(t)
            # HLS→RGB変換式に従い連続的に計算
            res[(t < 1/6)] = p[(t < 1/6)] + (q[(t < 1/6)] - p[(t < 1/6)]) * 6 * t[(t < 1/6)]
            res[(1/6 <= t) & (t < 1/2)] = q[(1/6 <= t) & (t < 1/2)]
            res[(1/2 <= t) & (t < 2/3)] = p[(1/2 <= t) & (t < 2/3)] + (q[(1/2 <= t) & (t < 2/3)] - p[(1/2 <= t) & (t < 2/3)]) * (4 - 6 * t[(1/2 <= t) & (t < 2/3)])
            res[(2/3 <= t)] = p[(2/3 <= t)]
            return res

        r = _hue2rgb(p, q, h + 1/3)
        g = _hue2rgb(p, q, h)
        b = _hue2rgb(p, q, h - 1/3)

        rgb = np.stack([r, g, b], axis=2)
        
        return np.clip(rgb, 0, 1).astype(np.float32)
    
    

    
    

        
    # --- 1. sRGB ⇄ 線形変換 ---
    def to_linear(self, x):
        """sRGB to linear"""
        x = np.clip(x, 0.0, 1.0)
        m = x <= 0.04045
        y = np.empty_like(x)
        y[m]      = x[m] / 12.92
        y[~m]     = ((x[~m] + 0.055) / 1.055) ** 2.4
        return y

    def to_srgb(self, x):
        """linear to sRGB"""
        x = np.clip(x, 0.0, 1.0)
        m = x <= 0.0031308
        y = np.empty_like(x)
        y[m]      = x[m] * 12.92
        y[~m]     = 1.055 * (x[~m] ** (1.0 / 2.4)) - 0.055
        return y
    

    
        
    
    def _adjustment_tone(
        self,
        linear_rgb_image: np.ndarray,
        exposure: float = 0.0,
        contrast: int = 0,
        shadow: int = 0,
        highlight: int = 0,
        black: int = 0,
        white: int = 0,
        mask: np.ndarray = None,
    ) -> np.ndarray:
        

        

        lut_np = self._create_tone_lut_from_params(exposure=exposure, 
                                                contrast=contrast, 
                                                shadow=shadow, 
                                                highlight=highlight, 
                                                black=black, 
                                                white=white,
                                                dtype=np.float32)

        linear_rgb_image *= 65535.0
        if mask is not None:
            linear_rgb_image[mask] = lut_np[linear_rgb_image[mask].astype(np.uint16)]
        else:
            linear_rgb_image = lut_np[linear_rgb_image.astype(np.uint16)]

        return (linear_rgb_image / 65535.0).astype(np.float32)

    

        

    
    
    
    def _adjustment_tone_curve(self, linear_rgb_image: np.ndarray, curve: np.ndarray, mask: np.ndarray = None):
        


        
        linear_rgb_image *= 65535.0
        if mask is not None:
            linear_rgb_image[mask] = curve[linear_rgb_image[mask].astype(np.uint16)]
        else:
            linear_rgb_image = curve[linear_rgb_image.astype(np.uint16)]
        
        return (linear_rgb_image / 65535.0).astype(np.float32)

    

    def _adjustment_hls_hue_tone_curve(self, hls_image: np.ndarray, curve: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        

        hls_image = hls_image.copy() * 65535.0

        # 色相チャンネル（H）に調整を適用
        if mask is not None:
            hls_image[mask, 0] = curve[hls_image[mask ,0].astype(np.uint16)]
        else:
            hls_image[:,:,0] = curve[hls_image[:,:,0].astype(np.uint16)]
        

        return (hls_image / 65535.0).astype(np.float32)

        
    
        
        
    def _adjustment_hls_saturation_tone_curve(self, hls_image: np.ndarray, curve: np.ndarray, mask: np.ndarray = None):
        
        # 彩度チャンネル（S）に調整を適用
        hls_image = hls_image.copy() * 65535.0

        if mask is not None:
            hue = hls_image[mask, 0].astype(np.uint16)
            
            hls_image[mask, 2] = np.clip(hls_image[mask, 2] * curve[hue] / 65535, 0, 65535)
        else:
            hue = hls_image[:,:,0].astype(np.uint16)
            hls_image[:,:,2] = np.clip(hls_image[:,:,2] * curve[hue] / 65535, 0, 65535)
        

        return (hls_image / 65535.0).astype(np.float32)

    
        
    
    def _adjustment_hls_lightness_tone_curve(self, hls_image: np.ndarray, curve: np.ndarray, mask: np.ndarray = None):
        

        hls_image = hls_image.copy() * 65535.0

        # 彩度に調整を適用

        if mask is not None:
            hue = hls_image[mask, 0].astype(np.uint16)
            hls_image[mask, 1] = np.clip(hls_image[mask, 1] * curve[hue] / 65535, 0, 65535)
        else:
            hue = hls_image[:,:,0].astype(np.uint16)
            hls_image[:,:,1] = np.clip(hls_image[:,:,1] * curve[hue] / 65535, 0, 65535)


        return (hls_image / 65535.0).astype(np.float32)

        

