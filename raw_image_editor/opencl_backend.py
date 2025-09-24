# raw_image_editor.opencl_backend.py

import numpy as np
import photo_metadata
from typing import Union, Any, Tuple, Dict, Optional
import copy
from scipy.ndimage import uniform_filter
from scipy.stats import skew, entropy
from .base import RAWImageEditorBase
import pyopencl as cl
import pyopencl.array as cl_array

# グローバル変数（init時に設定される）
ctx = None


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


class OpenCLCLAHE:
    """
    OpenCLを使用したCLAHE（Contrast Limited Adaptive Histogram Equalization）実装
    RGB画像の輝度ベースCLAHEを高速化
    """
    def __init__(self, context=None, queue=None):
        """
        OpenCL環境を初期化
        Args:
            context: OpenCLコンテキスト（Noneの場合は自動作成）
            queue: OpenCLコマンドキュー（Noneの場合は自動作成）
        """
        
        self.context = context
        
    
        self.queue = queue
            
        # カーネルプログラムをビルド
        self._build_kernels()

    def _build_kernels(self):
        """
        OpenCLカーネルをビルド
        """
        kernel_source = """
        // 輝度計算カーネル（BT.709）
        __kernel void compute_luminance(
            __global const float* rgb,
            __global float* luminance,
            const int width,
            const int height
        ) {
            int gid = get_global_id(0);
            if (gid >= width * height) return;
            
            int pixel_idx = gid * 3;
            float r = rgb[pixel_idx];
            float g = rgb[pixel_idx + 1];
            float b = rgb[pixel_idx + 2];
            
            luminance[gid] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }
        
        // タイルごとのヒストグラム計算カーネル
        __kernel void compute_tile_histogram(
            __global const float* luminance,
            __global float* histograms,
            const int width,
            const int height,
            const int tile_grid_x,
            const int tile_grid_y,
            const int num_bins
        ) {
            int tile_i = get_global_id(0);  // タイルのy座標
            int tile_j = get_global_id(1);  // タイルのx座標
            
            if (tile_i >= tile_grid_y || tile_j >= tile_grid_x) return;
            
            int tile_h = height / tile_grid_y;
            int tile_w = width / tile_grid_x;
            
            // タイル境界計算
            int y0 = tile_i * tile_h;
            int y1 = (tile_i < tile_grid_y - 1) ? y0 + tile_h : height;
            int x0 = tile_j * tile_w;
            int x1 = (tile_j < tile_grid_x - 1) ? x0 + tile_w : width;
            
            // ヒストグラム初期化
            int hist_offset = (tile_i * tile_grid_x + tile_j) * num_bins;
            for (int i = 0; i < num_bins; i++) {
                histograms[hist_offset + i] = 0.0f;
            }
            
            // ヒストグラム計算
            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    float val = luminance[y * width + x];
                    int bin_idx = (int)(val * (num_bins - 1));
                    bin_idx = clamp(bin_idx, 0, num_bins - 1);
                    histograms[hist_offset + bin_idx] += 1.0f;
                }
            }
        }
        
        // ヒストグラムクリップ＆CDF計算カーネル
        __kernel void compute_cdf(
            __global float* histograms,
            __global float* cdfs,
            const int tile_grid_x,
            const int tile_grid_y,
            const int num_bins,
            const float clip_limit,
            const int tile_area
        ) {
            int tile_i = get_global_id(0);
            int tile_j = get_global_id(1);
            
            if (tile_i >= tile_grid_y || tile_j >= tile_grid_x) return;
            
            int hist_offset = (tile_i * tile_grid_x + tile_j) * num_bins;
            float max_per_bin = clip_limit * tile_area;
            
            // クリップ＆再分配
            float excess = 0.0f;
            for (int i = 0; i < num_bins; i++) {
                if (histograms[hist_offset + i] > max_per_bin) {
                    excess += histograms[hist_offset + i] - max_per_bin;
                    histograms[hist_offset + i] = max_per_bin;
                }
            }
            
            float redistribute = excess / num_bins;
            for (int i = 0; i < num_bins; i++) {
                histograms[hist_offset + i] += redistribute;
            }
            
            // CDF計算
            cdfs[hist_offset] = histograms[hist_offset];
            for (int i = 1; i < num_bins; i++) {
                cdfs[hist_offset + i] = cdfs[hist_offset + i - 1] + histograms[hist_offset + i];
            }
            
            // 正規化
            float total = cdfs[hist_offset + num_bins - 1];
            if (total > 0.0f) {
                for (int i = 0; i < num_bins; i++) {
                    cdfs[hist_offset + i] /= total;
                }
            }
        }
        
        // 双線形補間カーネル（CLAHE適用）
        __kernel void apply_clahe(
            __global const float* luminance,
            __global const float* cdfs,
            __global float* output_luminance,
            const int width,
            const int height,
            const int tile_grid_x,
            const int tile_grid_y,
            const int num_bins
        ) {
            int gid = get_global_id(0);
            if (gid >= width * height) return;
            
            int y = gid / width;
            int x = gid % width;
            
            float val = luminance[gid];
            int bin_idx = (int)(val * (num_bins - 1));
            bin_idx = clamp(bin_idx, 0, num_bins - 1);
            
            float tile_h = (float)height / tile_grid_y;
            float tile_w = (float)width / tile_grid_x;
            
            // タイル内位置
            float fy = y / tile_h - 0.5f;
            float fx = x / tile_w - 0.5f;
            int i = (int)floor(fy);
            int j = (int)floor(fx);
            float dy = fy - i;
            float dx = fx - j;
            
            // 周囲4タイルからCDF値取得（境界処理込み）
            int i0 = clamp(i, 0, tile_grid_y - 1);
            int i1 = clamp(i + 1, 0, tile_grid_y - 1);
            int j0 = clamp(j, 0, tile_grid_x - 1);
            int j1 = clamp(j + 1, 0, tile_grid_x - 1);
            
            float v00 = cdfs[(i0 * tile_grid_x + j0) * num_bins + bin_idx];
            float v01 = cdfs[(i0 * tile_grid_x + j1) * num_bins + bin_idx];
            float v10 = cdfs[(i1 * tile_grid_x + j0) * num_bins + bin_idx];
            float v11 = cdfs[(i1 * tile_grid_x + j1) * num_bins + bin_idx];
            
            // 双線形補間
            output_luminance[gid] = (1.0f - dy) * (1.0f - dx) * v00 +
                                (1.0f - dy) * dx * v01 +
                                dy * (1.0f - dx) * v10 +
                                dy * dx * v11;
        }
        
        // RGB色保持適用カーネル
        __kernel void apply_luminance_ratio(
            __global const float* original_rgb,
            __global const float* original_luminance,
            __global const float* enhanced_luminance,
            __global float* output_rgb,
            const int width,
            const int height
        ) {
            int gid = get_global_id(0);
            if (gid >= width * height) return;
            
            float orig_lum = original_luminance[gid];
            float enh_lum = enhanced_luminance[gid];
            
            // ゼロ除算防止
            float ratio = enh_lum / (orig_lum + 1e-6f);
            
            int pixel_idx = gid * 3;
            output_rgb[pixel_idx] = clamp(original_rgb[pixel_idx] * ratio, 0.0f, 1.0f);
            output_rgb[pixel_idx + 1] = clamp(original_rgb[pixel_idx + 1] * ratio, 0.0f, 1.0f);
            output_rgb[pixel_idx + 2] = clamp(original_rgb[pixel_idx + 2] * ratio, 0.0f, 1.0f);
        }
        """
        
        self.program = cl.Program(self.context, kernel_source).build()



    def clahe_rgb(self, rgb_img, tile_grid_y=8, tile_grid_x=8, clip_limit=0.003, num_bins=512):
        """
        RGB画像に対してCLAHEを適用（輝度ベース、色保持）
        Args:
            rgb_img (np.ndarray): 入力RGB画像（float32, [0.0, 1.0]）, shape=(H, W, 3)
            tile_grid_y (int): タイルの縦分割数
            tile_grid_x (int): タイルの横分割数
            clip_limit (float): クリップ制限率
            num_bins (int): ヒストグラムビン数
        Returns:
            np.ndarray: 補正後RGB画像（float32, [0.0, 1.0]）, shape=(H, W, 3)
        """
        H, W = rgb_img.shape[:2]
        tile_h = H // tile_grid_y
        tile_w = W // tile_grid_x
        tile_area = tile_h * tile_w
        
        # 入力データをGPUにコピー
        rgb_flat = rgb_img.astype(np.float32).flatten()
        rgb_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=rgb_flat)
        
        # バッファ作成
        luminance_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=H * W * 4)
        enhanced_luminance_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=H * W * 4)
        histograms_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=tile_grid_y * tile_grid_x * num_bins * 4)
        cdfs_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=tile_grid_y * tile_grid_x * num_bins * 4)
        output_rgb_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=H * W * 3 * 4)
        
        # 1. 輝度計算
        self.program.compute_luminance(
            self.queue, (H * W,), None,
            rgb_buffer, luminance_buffer,
            np.int32(W), np.int32(H)
        )
        
        # 2. タイルごとのヒストグラム計算
        self.program.compute_tile_histogram(
            self.queue, (tile_grid_y, tile_grid_x), None,
            luminance_buffer, histograms_buffer,
            np.int32(W), np.int32(H),
            np.int32(tile_grid_x), np.int32(tile_grid_y),
            np.int32(num_bins)
        )
        
        # 3. CDF計算（クリップ＆正規化）
        self.program.compute_cdf(
            self.queue, (tile_grid_y, tile_grid_x), None,
            histograms_buffer, cdfs_buffer,
            np.int32(tile_grid_x), np.int32(tile_grid_y),
            np.int32(num_bins), np.float32(clip_limit),
            np.int32(tile_area)
        )
        
        # 4. CLAHE適用（双線形補間）
        self.program.apply_clahe(
            self.queue, (H * W,), None,
            luminance_buffer, cdfs_buffer, enhanced_luminance_buffer,
            np.int32(W), np.int32(H),
            np.int32(tile_grid_x), np.int32(tile_grid_y),
            np.int32(num_bins)
        )
        
        # 5. RGB色保持適用
        self.program.apply_luminance_ratio(
            self.queue, (H * W,), None,
            rgb_buffer, luminance_buffer, enhanced_luminance_buffer,
            output_rgb_buffer,
            np.int32(W), np.int32(H)
        )
        
        # 結果をCPUにコピー
        output_rgb = np.empty(H * W * 3, dtype=np.float32)
        cl.enqueue_copy(self.queue, output_rgb, output_rgb_buffer)
        
        return output_rgb.reshape(H, W, 3)

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


class RAWImageEditorOpenCL(RAWImageEditorBase):
    # Class variables for OpenCL resources
    ctx: cl.Context = None
    queue: cl.CommandQueue = None
    clahe: OpenCLCLAHE = None
    clip_0_1_kernel: cl.Kernel = None
    tone_curve_lut_kernel: cl.Kernel = None
    rgb_to_hls_kernel: cl.Kernel = None
    hls_to_rgb_kernel: cl.Kernel = None
    to_linear_kernel: cl.Kernel = None
    to_srgb_kernel: cl.Kernel = None
    tone_curve_by_hue_kernel: cl.Kernel = None
    white_balance_kernel: cl.Kernel = None
    
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

        RAWImageEditorのOpenCLバックエンド


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
            self.image_array = apply_gamma(self.image_array, gamma=gamma)
           
        
        if apply_clahe and is_raw:
            clip = 0.002
            print(f"clip: {clip}")
            self.image_array = self.clahe.clahe_rgb(
                                         self.image_array, 
                                         tile_grid_y=8,
                                         tile_grid_x=8, 
                                         clip_limit=clip,
                                         num_bins=512
                                         )
            
        self.initial_image_array = self.image_array.copy()
        
        

        self.full_mask = cl_array.to_device(self.queue, np.ones((self.height * self.width,), dtype=np.uint8))
        self.hue_channels = cl_array.to_device(self.queue, np.array([0], dtype=np.int32))
        self.full_channels = cl_array.to_device(self.queue, np.array([0, 1, 2], dtype=np.int32))

        self.num_pixels: int = self.height * self.width
        self.num_elements: int = self.num_pixels * 3
        



    def copy(self):
        return copy.copy(self)


    def apply_adjustments(self):
        """すべての編集を適用する関数"""
        
        # numpy -> OpenCL array
        self.image_array = np.clip(self.image_array, 0.0, 1.0)
        self.image_array_cl = cl_array.to_device(self.queue, self.image_array.flatten().astype(np.float32))
        
        # sRGB -> Linear (in-place)
        self.to_linear(self.image_array_cl)

        # 1. メインのトーン調整（露光量、コントラストなど）
        self._adjustment_tone(
            self.image_array_cl,
            exposure=self.main_adjustments_values["exposure"],
            contrast=self.main_adjustments_values["contrast"],
            shadow=self.main_adjustments_values["shadow"],
            highlight=self.main_adjustments_values["highlight"],
            black=self.main_adjustments_values["black"],
            white=self.main_adjustments_values["white"],
            mask=None
        )

        self._clip_0_1(self.image_array_cl)

        # 2. メインの輝度トーンカーブ調整
        self._adjustment_tone_curve(
            self.image_array_cl,
            self.main_adjustments_values["brightness_tone_curve"]
        )

        # 3. Linear -> sRGB (in-place)
        self.to_srgb(self.image_array_cl)
        
        # 4. ホワイトバランス調整
        self._adjustment_whitebalance(
            self.image_array_cl,
            temperature=self.main_adjustments_values["wb_temperature"],
            tint=self.main_adjustments_values["wb_tint"]
        )
        
        # 5. RGB -> HLS (in-place)
        self.rgb_to_hls_image(self.image_array_cl)
        self._clip_0_1(self.image_array_cl)
        
        # 6. メインのHLSトーンカーブ調整
        self._adjustment_hls_hue_tone_curve(
            self.image_array_cl,
            self.main_adjustments_values["hue_tone_curve"]
        )
        self._adjustment_hls_saturation_tone_curve(
            self.image_array_cl,
            self.main_adjustments_values["saturation_tone_curve"]
        )
        self._adjustment_hls_lightness_tone_curve(
            self.image_array_cl,
            self.main_adjustments_values["lightness_tone_curve"]
        )

        # 7. 各マスク領域に対するHLSトーンカーブ調整
        for mask_data in self.mask_adjustments_values.values():
            bool_mask = mask_data["ndarray"] > mask_data["mask_range_value"]
            
            self._adjustment_tone_curve(
                self.image_array_cl,
                mask_data["brightness_tone_curve"],
                mask=bool_mask
            )
            self._adjustment_hls_hue_tone_curve(
                self.image_array_cl,
                mask_data["hue_tone_curve"],
                mask=bool_mask
            )
            self._adjustment_hls_saturation_tone_curve(
                self.image_array_cl,
                mask_data["saturation_tone_curve"],
                mask=bool_mask
            )
            self._adjustment_hls_lightness_tone_curve(
                self.image_array_cl,
                mask_data["lightness_tone_curve"],
                mask=bool_mask
            )

        # 8. HLS -> RGB (in-place)
        self.hls_to_rgb_image(self.image_array_cl)

        # 9. マスクごとのRGB空間での調整
        for mask_data in self.mask_adjustments_values.values():
            bool_mask = mask_data["ndarray"] > mask_data["mask_range_value"]
            self._adjustment_tone(
                self.image_array_cl,
                exposure=mask_data["exposure"],
                contrast=mask_data["contrast"],
                shadow=mask_data["shadow"],
                highlight=mask_data["highlight"],
                black=mask_data["black"],
                white=mask_data["white"],
                mask=bool_mask)
            self._adjustment_whitebalance(
                self.image_array_cl,
                temperature=mask_data["wb_temperature"],
                tint=mask_data["wb_tint"],
                mask=bool_mask
            )

        # 10. 最終的なクリップ
        self._clip_0_1(self.image_array_cl)
        
        # OpenCL -> numpy
        self.image_array = self.image_array_cl.get().reshape(self.height, self.width, 3)
        


    def _clip_0_1(self, img_cl: cl_array.Array):
        """OpenCLで0-1にクリップする"""
        
        self.clip_0_1_kernel(self.queue, (self.num_elements,), None, img_cl.data, np.int32(self.num_elements))
        self.queue.finish()
        


    def _adjustment_whitebalance(self, rgb_image_cl: cl_array.Array, temperature: int, tint: int, mask: np.ndarray = None):
        
        # チャンネル毎のゲイン計算
        r_gain = np.float32(1.0 + 0.5 * (temperature / 100))
        b_gain = np.float32(1.0 - 0.5 * (temperature / 100))
        g_gain = np.float32(1.0 - 0.25 * (tint / 100))
        
        # Create mask buffer
        
        
        if mask is not None:
            mask_cl = cl_array.to_device(self.queue, mask.flatten().astype(np.uint8))
        else:
            mask_cl = self.full_mask
        
        self.white_balance_kernel(self.queue, (self.num_pixels,), None,
                                 rgb_image_cl.data, mask_cl.data, np.int32(self.num_pixels),
                                 r_gain, g_gain, b_gain)
        self.queue.finish()
        
        


    def rgb_to_hls_image(self, rgb_image_cl: cl_array.Array):
        """
        RGB画像をHLS画像に変換する関数
        """
        
        
        
        self.rgb_to_hls_kernel(self.queue, (self.num_pixels,), None, rgb_image_cl.data, np.int32(self.num_pixels))
        self.queue.finish()
        
        


    def hls_to_rgb_image(self, hls_image_cl: cl_array.Array):
        """
        HLS画像をRGB画像に変換する関数
        """
        
        
        
        self.hls_to_rgb_kernel(self.queue, (self.num_pixels,), None, hls_image_cl.data, np.int32(self.num_pixels))
        self.queue.finish()
        
        


    # --- 1. sRGB ⇄ 線形変換 ---
    def to_linear(self, img_cl: cl_array.Array):
        """sRGB to linear"""
        
        
        self.to_linear_kernel(self.queue, (self.num_elements,), None, img_cl.data, np.int32(self.num_elements))
        self.queue.finish()
        

    def to_srgb(self, img_cl: cl_array.Array):
        """linear to sRGB"""
        
        
        self.to_srgb_kernel(self.queue, (self.num_elements,), None, img_cl.data, np.int32(self.num_elements))
        self.queue.finish()
        


    def _adjustment_tone(
        self,
        img_cl: cl_array.Array,
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
        # LUTをGPUに転送
        lut_cl = cl_array.to_device(self.queue, lut_np)
        
        # マスクを準備
        if mask is not None:
            mask_cl = cl_array.to_device(self.queue, mask.flatten().astype(np.uint8))
        else:
            mask_cl = self.full_mask
        
        # カーネルを実行して、元の画像バッファにLUTを適用
        self.tone_curve_lut_kernel(self.queue, (self.num_pixels,), None,
                                  img_cl.data, lut_cl.data, mask_cl.data,
                                  self.full_channels.data, np.int32(3), np.int32(self.num_pixels))
        self.queue.finish()


    def _adjustment_tone_curve(self, linear_rgb_image: cl_array.Array, curve: np.ndarray, mask: np.ndarray = None):
        
        # マスクを準備
        if mask is not None:
            mask_cl = cl_array.to_device(self.queue, mask.flatten().astype(np.uint8))
        else:
            mask_cl = self.full_mask
        
        # カーブ（LUT）をGPUに転送
        curve_cl = cl_array.to_device(self.queue, curve.astype(np.float32))
        
        # トーンカーブを適用
        self.tone_curve_lut_kernel(self.queue, (self.num_pixels,), None,
                                  linear_rgb_image.data, curve_cl.data, mask_cl.data,
                                  self.full_channels.data, np.int32(3), np.int32(self.num_pixels))
        self.queue.finish()
        


    def _adjustment_hls_hue_tone_curve(self, hls_image: cl_array.Array, curve: np.ndarray, mask: np.ndarray = None):
        
        # マスクを準備
        if mask is not None:
            mask_cl = cl_array.to_device(self.queue, mask.flatten().astype(np.uint8))
        else:
            mask_cl = self.full_mask
        
        # カーブ（LUT）をGPUに転送
        curve_cl = cl_array.to_device(self.queue, curve.astype(np.float32))
        
        # Hチャンネルにトーンカーブを適用
        self.tone_curve_lut_kernel(self.queue, (self.num_pixels,), None,
                                  hls_image.data, curve_cl.data, mask_cl.data,
                                  self.hue_channels.data, np.int32(1), np.int32(self.num_pixels))
        self.queue.finish()
        


    def _adjustment_hls_saturation_tone_curve(self, hls_image: cl_array.Array, curve: np.ndarray, mask: np.ndarray = None):
        
        # マスクを準備
        if mask is not None:
            mask_cl = cl_array.to_device(self.queue, mask.flatten().astype(np.uint8))
        else:
            mask_cl = self.full_mask
        
        # カーブ（LUT）をGPUに転送
        curve_cl = cl_array.to_device(self.queue, curve.astype(np.float32))
        
        # 彩度チャンネルにトーンカーブを適用
        self.tone_curve_by_hue_kernel(self.queue, (self.num_pixels,), None,
                                      hls_image.data, curve_cl.data, mask_cl.data,
                                      np.int32(0), np.int32(2), np.int32(self.num_pixels))
        self.queue.finish()
        


    def _adjustment_hls_lightness_tone_curve(self, hls_image: cl_array.Array, curve: np.ndarray, mask: np.ndarray = None):
        
        # マスクを準備
        if mask is not None:
            mask_cl = cl_array.to_device(self.queue, mask.flatten().astype(np.uint8))
        else:
            mask_cl = self.full_mask
        
        # カーブ（LUT）をGPUに転送
        curve_cl = cl_array.to_device(self.queue, curve.astype(np.float32))
        
        # 輝度チャンネルにトーンカーブを適用
        self.tone_curve_by_hue_kernel(self.queue, (self.num_pixels,), None,
                                      hls_image.data, curve_cl.data, mask_cl.data,
                                      np.int32(0), np.int32(1), np.int32(self.num_pixels))
        self.queue.finish()
        