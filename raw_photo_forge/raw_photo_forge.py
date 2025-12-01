"RAW現像ソフトメインUI"

import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

    
import json
import copy
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import traceback
import threading
import time
import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox, Canvas, Frame, Label, Button, Checkbutton, BooleanVar
from tkinter import Toplevel, StringVar, IntVar, DoubleVar
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from scipy.interpolate import PchipInterpolator


try:
    import raw_image_editor
    
    RAW_EDITOR_AVAILABLE = True
except Exception:
    traceback.print_exc()
    RAW_EDITOR_AVAILABLE = False
    print("Warning: raw_image_editor or photo_metadata not available")

# OKLCH変換用の定数と関数
oklab_to_lms_m = np.array([
    [1.0, 0.3963377774, 0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480]
])

lms_to_srgb = np.array([
    [4.0767245293, -3.3072168827, 0.2307590544],
    [-1.2681437731, 2.6093323231, -0.3411344290],
    [-0.0041119885, -0.7034763098, 1.7068625689]
])

def oklab_to_lms(lab):
    lms_ = np.dot(oklab_to_lms_m, lab)
    return lms_ ** 3

def lms_to_linear_srgb(lms):
    return np.dot(lms_to_srgb, lms)

def oklch_to_rgb(l, c, h):
    h_rad = h * 2 * np.pi
    a = c * np.cos(h_rad)
    b = c * np.sin(h_rad)
    
    lms = oklab_to_lms(np.array([l, a, b]))
    
    rgb_linear = lms_to_linear_srgb(lms)
    
    # np.powerに負の値が入らないようにクリップ
    rgb_linear_clipped = np.maximum(0, rgb_linear)
    
    # sRGBにガンマ補正
    rgb = np.where(
        rgb_linear_clipped <= 0.0031308,
        rgb_linear_clipped * 12.92,
        1.055 * np.power(rgb_linear_clipped, 1.0/2.4) - 0.055
    )
    return np.clip(rgb, 0, 1)

def oklch_to_rgb_vectorized(L, C, H):
    """OKLCHからsRGBに変換するベクトル化された関数。
    L, C, Hは同じ形状のNumPy配列を想定。
    """
    h_rad = H * (2 * np.pi)
    a = C * np.cos(h_rad)
    b = C * np.sin(h_rad)

    # (height, width, 3) のLab配列を作成
    lab = np.stack([L, a, b], axis=-1)

    # Oklab -> LMS
    # (h, w, 3) @ (3, 3) -> (h, w, 3)
    lms_ = np.matmul(lab, oklab_to_lms_m.T)
    lms = lms_ ** 3

    # LMS -> linear sRGB
    # (h, w, 3) @ (3, 3) -> (h, w, 3)
    rgb_linear = np.matmul(lms, lms_to_srgb.T)

    # sRGBにガンマ補正
    # np.powerに負の値が入らないようにクリップ
    rgb_linear_clipped = np.maximum(0, rgb_linear)

    rgb = np.where(
        rgb_linear_clipped <= 0.0031308,
        rgb_linear_clipped * 12.92,
        1.055 * np.power(rgb_linear_clipped, 1.0/2.4) - 0.055
    )
    return np.clip(rgb, 0, 1)

@dataclass
class EditParameters:
    """編集パラメータを管理するデータクラス"""
    # 露出パラメータ
    exposure: float = 0.0
    contrast: int = 0
    shadow: int = 0
    highlight: int = 0
    black: int = 0
    white: int = 0
    
    # ホワイトバランス
    wb_temperature: int = 0
    wb_tint: int = 0
    
    # トーンカーブ（制御点として保存）
    brightness_curve_points: List[Tuple[float, float]] = None
    oklch_h_curve_points: List[Tuple[float, float]] = None
    oklch_c_curve_points: List[Tuple[float, float]] = None
    oklch_l_curve_points: List[Tuple[float, float]] = None

    # 周辺減光
    vignette: int = 0

    # マスク範囲
    mask_range: float = 0.0
    
    
    def __post_init__(self):
        if self.brightness_curve_points is None:
            self.brightness_curve_points = [(0, 0), (65535, 65535)]
        if self.oklch_h_curve_points is None:
            self.oklch_h_curve_points = [(0, 0), (65535, 65535)]
        if self.oklch_c_curve_points is None:
            self.oklch_c_curve_points = [(0, 32767.5), (65535, 32767.5)]
        if self.oklch_l_curve_points is None:
            self.oklch_l_curve_points = [(0, 32767.5), (65535, 32767.5)]

@dataclass
class SettingsManager:
    """設定とパスを管理するデータクラス"""
    device_index: int = 0
    language: str = "日本語"
    preview_size: int = 1000
    dragging_preview_size: int = 300
    
    # パス関連（初期化後に設定）
    app_dir: Path = None
    settings_path: Path = None
    exiftool_path: str = None

    def __post_init__(self):
        """パスの初期化"""
        if getattr(sys, 'frozen', False):
            # exe化されている場合
            self.app_dir = Path(sys.executable).parent
            self.exiftool_path = str(self.app_dir / "exiftool_dir" / "exiftool.exe")
        else:
            # .pyで実行されている場合
            self.app_dir = Path(__file__).parent
            self.exiftool_path = "exiftool" # 環境変数PATHにあることを期待
        
        self.settings_path = self.app_dir / "settings.json"

    def load(self):
        """設定をJSONファイルから読み込む"""
        if not self.settings_path.exists():
            return
        try:
            with open(self.settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            
            # デバイス設定の読み込みと検証
            device_index = settings.get("device", self.device_index)
            
            if RAW_EDITOR_AVAILABLE and raw_image_editor.SLANGPY_AVAILABLE:
                try:
                    adapters = raw_image_editor.get_slangpy_adapters()
                    index = int(device_index)
                    if index < len(adapters):
                        self.device_index = index
                    else:
                        self.device_index = 0  # インデックスが範囲外
                except (ValueError, IndexError, Exception):
                    traceback.print_exc()
                    self.device_index = 0  # パースエラーやSlangpyエラー
            else:
                self.device_index = 0  # Slangpyが利用不可
            

            self.language = settings.get("language", self.language)
            self.preview_size = settings.get("preview_size", self.preview_size)
            self.dragging_preview_size = settings.get("dragging_preview_size", self.dragging_preview_size)
        except Exception as e:
            traceback.print_exc()
            print(f"設定ファイルの読み込みに失敗しました: {e}")

    def save(self):
        """現在の設定をJSONファイルに保存する"""
        settings = {
            "device": self.device_index,
            "language": self.language,
            "preview_size": self.preview_size,
            "dragging_preview_size": self.dragging_preview_size
        }
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            traceback.print_exc()
            print(f"設定の保存に失敗しました: {e}")
            raise e

class ToneCurveWidget:
    """matplotlibを使ったトーンカーブエディタ"""
    
    def __init__(self, parent, curve_type="brightness", main_window=None):
        self.parent = parent
        self.main_window = main_window
        self.curve_type = curve_type
        
        # 正方形を保つためのフレーム
        self.container_frame = Frame(parent)
        self.container_frame.pack(fill=tk.BOTH, expand=True)
        self.container_frame.bind('<Configure>', self.on_container_resize)
        
        # matplotlibの図を作成（正方形）
        self.fig = Figure(figsize=(4, 4), facecolor='white')
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.container_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        
        self.histogram_data = None  # ヒストグラムデータを保持
        self.ax = self.fig.add_subplot(111)
        self.setup_plot()
        
        # 初期制御点
        if curve_type == "brightness" or curve_type == "oklch_h":
            self.control_points = [(0, 0), (65535, 65535)]
        else:  # oklch_c, oklch_l
            self.control_points = [(0, 32767.5), (65535, 32767.5)]
            
        self.selected_point = None
        self.dragging = False
        
        # イベント接続
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.update_plot()
        self.update_canvas_position()
    
    def on_container_resize(self, event):
        """コンテナリサイズ時の処理"""
        self.update_canvas_position()
    
    def update_canvas_position(self):
        """キャンバスの位置を更新（常に正方形を維持）"""
        container_width = self.container_frame.winfo_width()
        container_height = self.container_frame.winfo_height()
        
        if container_width > 1 and container_height > 1:  # キャンバスが初期化済み
            # 正方形のサイズを計算（小さい方に合わせる）
            square_size = min(container_width, container_height)
            
            # 中央に配置
            x = (container_width - square_size) // 2
            y = (container_height - square_size) // 2
            
            self.canvas_widget.place(x=x, y=y, width=square_size, height=square_size)
    
    def setup_plot(self):
        """プロットの基本設定"""
        self.ax.clear()
        self.ax.set_xlim(0, 65535)
        self.ax.set_ylim(0, 65535)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # 背景の設定
        if self.curve_type == "brightness":
            # グレーのグラデーション背景
            gradient = np.linspace(0, 1, 256).reshape(256, 1)[::-1]
            gradient = np.repeat(gradient, 256, axis=1)
            self.ax.imshow(gradient, extent=[0, 65535, 0, 65535], cmap='gray', alpha=0.7, aspect='auto')

            # ヒストグラムの描画
            if self.histogram_data:
                x_values = np.arange(256) * (65535 / 255.0)
                
                # ヒストグラムの最大値を見つけて正規化
                max_hist_val = 1.0 # ゼロ除算を避ける
                for key in ['white', 'r', 'g', 'b']:
                    if key in self.histogram_data and self.histogram_data[key] is not None:
                        max_val = np.max(self.histogram_data[key])
                        if max_val > max_hist_val:
                            max_hist_val = max_val
                
                scale = 65535 / max_hist_val
                
                # ヒストグラムを線で描画
                if 'white' in self.histogram_data and self.histogram_data['white'] is not None:
                    self.ax.plot(x_values, self.histogram_data['white'] * scale, color='white', alpha=0.6, linewidth=3)
                if 'r' in self.histogram_data and self.histogram_data['r'] is not None:
                    self.ax.plot(x_values, self.histogram_data['r'] * scale, color='red', alpha=0.6, linewidth=1)
                if 'g' in self.histogram_data and self.histogram_data['g'] is not None:
                    self.ax.plot(x_values, self.histogram_data['g'] * scale, color='green', alpha=0.6, linewidth=1)
                if 'b' in self.histogram_data and self.histogram_data['b'] is not None:
                    self.ax.plot(x_values, self.histogram_data['b'] * scale, color='blue', alpha=0.6, linewidth=1)
            
        elif self.curve_type == "oklch_h":
            # 色相のグラデーション背景とカラーライン
            x = np.linspace(0, 1, 256)
            y = np.linspace(0, 1, 256)
            X, Y = np.meshgrid(x, y)
            
            L = np.full_like(X, 0.75)
            C = np.full_like(X, 0.2)
            H = X
            
            colors = oklch_to_rgb_vectorized(L, C, H)

            self.ax.imshow(colors, extent=[0, 65535, 0, 65535], aspect='auto', alpha=0.7, origin='lower')
            
            # カラーライン
            y_pos = np.linspace(0, 65535, 7)
            colors_line = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'red']
            for y, color in zip(y_pos, colors_line):
                self.ax.axhline(y=y, color=color, linewidth=2, alpha=0.8)

        elif self.curve_type == "oklch_c":
            x = np.linspace(0, 1, 256)
            y = np.linspace(0, 1, 256)
            X, Y = np.meshgrid(x, y)
            
            L = np.full_like(X, 0.75)
            C = Y * 0.4 
            H = X
            
            colors = oklch_to_rgb_vectorized(L, C, H)

            self.ax.imshow(colors, extent=[0, 65535, 0, 65535], aspect='auto', alpha=0.7, origin='lower')

        elif self.curve_type == "oklch_l":
            x = np.linspace(0, 1, 256)
            y = np.linspace(0, 1, 256)
            X, Y = np.meshgrid(x, y)
            
            L = Y
            C = np.full_like(X, 0.2)
            H = X
            
            colors = oklch_to_rgb_vectorized(L, C, H)
            
            self.ax.imshow(colors, extent=[0, 65535, 0, 65535], aspect='auto', alpha=0.7, origin='lower')
    
    def update_plot(self):
        """プロットを更新"""
        self.setup_plot()
        
        if len(self.control_points) >= 2:
            # 制御点からカーブを生成
            points = sorted(self.control_points)
            x_points = [p[0] for p in points]
            y_points = [p[1] for p in points]
            
            # PchipInterpolatorでスムーズなカーブを作成
            if len(points) >= 2:
                x_curve = np.linspace(0, 65535, 65536)
                interp = PchipInterpolator(x_points, y_points)
                y_curve = interp(x_curve)
                y_curve = np.clip(y_curve, 0, 65535)
                
                # カーブをプロット
                self.ax.plot(x_curve, y_curve, 'b-', linewidth=2)
                
                # 直線でない場合は参照線を表示
                is_straight = self.is_straight_line()
                if not is_straight:
                    if self.curve_type == "brightness" or self.curve_type == "oklch_h":
                        self.ax.plot([0, 65535], [0, 65535], 'b--', alpha=0.5, linewidth=1)
                    else:  # oklch_c, oklch_l
                        self.ax.plot([0, 65535], [32767.5, 32767.5], 'b--', alpha=0.5, linewidth=1)
        
        # 制御点をプロット
        for i, (x, y) in enumerate(self.control_points):
            self.ax.plot(x, y, 'ro', markeredgecolor='black', markeredgewidth=2, markersize=8)
        
        self.canvas.draw()
    
    def is_straight_line(self):
        """直線かどうかを判定"""
        if len(self.control_points) <= 2:
            if self.curve_type == "brightness" or self.curve_type == "oklch_h":
                return self.control_points == [(0, 0), (65535, 65535)]
            else:  # oklch_c, oklch_l
                return self.control_points == [(0, 32767.5), (65535, 32767.5)]
        return False
    
    def on_press(self, event):
        """マウスクリック処理"""
        if event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        # 近くの制御点を検索
        nearest_point = None
        min_distance = float('inf')
        for i, (px, py) in enumerate(self.control_points):
            distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_point = i
        
        # 右クリック：制御点削除
        if event.button == 3:  # 右クリック
            if min_distance < 2000 and len(self.control_points) > 2:
                # 最初と最後の点は削除できない
                if nearest_point != 0 and nearest_point != len(self.control_points) - 1:
                    del self.control_points[nearest_point]
                    self.update_plot()
                    self.emit_curve_changed()
            return
        
        # 左クリック：制御点選択または作成
        if event.button == 1:  # 左クリック
            if min_distance < 2000:
                self.selected_point = nearest_point
                self.dragging = True
            else:
                # 新しい制御点を作成
                new_point = (x, y)
                self.control_points.append(new_point)
                self.control_points.sort(key=lambda p: p[0])
                self.selected_point = self.control_points.index(new_point)
                self.dragging = True
                self.update_plot()
                self.emit_curve_changed()
    
    def on_motion(self, event):
        """マウス移動処理"""
        if not self.dragging or self.selected_point is None:
            return
        
        if event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        # 制御点を移動
        old_point = self.control_points[self.selected_point]
        
        # 最初と最後の点は横移動不可
        if self.selected_point == 0:
            x = 0
        elif self.selected_point == len(self.control_points) - 1:
            x = 65535
        
        # 範囲チェック
        x = max(0, min(65535, x))
        y = max(0, min(65535, y))
        
        self.control_points[self.selected_point] = (x, y)
        self.control_points.sort(key=lambda p: p[0])
        
        # ソート後の新しいインデックスを見つける
        self.selected_point = self.control_points.index((x, y))
        
        self.update_plot()
        self.emit_curve_changed()
    
    def on_release(self, event):
        """マウスリリース処理"""
        was_dragging = self.dragging
        self.dragging = False
        self.selected_point = None
        if was_dragging and self.main_window:
            self.main_window.update_image_display(force_full=True)
    
    def emit_curve_changed(self):
        """カーブ変更シグナルを発信"""
        if self.main_window:
            if self.curve_type == "brightness":
                self.main_window.on_brightness_curve_changed(self.control_points.copy())
            elif self.curve_type == "oklch_h":
                self.main_window.on_oklch_h_curve_changed(self.control_points.copy())
            elif self.curve_type == "oklch_c":
                self.main_window.on_oklch_c_curve_changed(self.control_points.copy())
            elif self.curve_type == "oklch_l":
                self.main_window.on_oklch_l_curve_changed(self.control_points.copy())

    def set_histogram(self, hist_data):
        """ヒストグラムデータを設定する"""
        self.histogram_data = hist_data
        # 明るさカーブの場合のみプロットを更新
        if self.curve_type == "brightness":
            self.update_plot()
    
    def set_control_points(self, points):
        """制御点を設定"""
        # プリセット読み込み時にJSON由来のリストのリストが渡されることがあるため、タプルのリストに変換する
        if points:
            self.control_points = [tuple(p) for p in points]
        else:
            # pointsがNoneや空リストの場合、デフォルトにリセットする
            if self.curve_type == "brightness" or self.curve_type == "oklch_h":
                self.control_points = [(0, 0), (65535, 65535)]
            else:  # oklch_c, oklch_l
                self.control_points = [(0, 32767.5), (65535, 32767.5)]
        self.update_plot()
    
    def reset_curve(self):
        """カーブをリセット"""
        if self.curve_type == "brightness" or self.curve_type == "oklch_h":
            self.control_points = [(0, 0), (65535, 65535)]
        else:  # oklch_c, oklch_l
            self.control_points = [(0, 32767.5), (65535, 32767.5)]
        self.update_plot()
        self.emit_curve_changed()
    
    def get_curve_array(self):
        """65536要素のカーブ配列を取得"""
        if len(self.control_points) < 2:
            return np.arange(65536, dtype=np.float32)
        
        points = sorted(self.control_points)
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        
        x_curve = np.arange(65536, dtype=np.float32)
        interp = PchipInterpolator(x_points, y_points)
        y_curve = interp(x_curve)
        y_curve = np.clip(y_curve, 0, 65535)
        
        return y_curve

class SliderWithSpinBox:
    """スライダーと数値入力ボックスを組み合わせたウィジェット"""
    
    def __init__(self, parent, min_val, max_val, step, *, dtype, decimals=0, initial_value=0, callback=None):
        self.parent = parent
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.dtype = dtype
        self.decimals = decimals
        self.callback = callback
        self.updating = False
        
        self.frame = Frame(parent)
        
        # スライダー
        self.var = DoubleVar()
        self.var.set(initial_value)
        
        self.slider = ttk.Scale(self.frame, from_=min_val, to=max_val, variable=self.var, 
                               orient=tk.HORIZONTAL, command=self.on_slider_changed)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # 数値入力ボックス
        if self.dtype is float:
            self.spinbox = ttk.Spinbox(self.frame, from_=min_val, to=max_val, increment=step,
                                      textvariable=self.var, width=10, format=f"%.{self.decimals}f")
        else: # int
            vcmd = (self.frame.register(self.validate_int_input), '%P')
            self.spinbox = ttk.Spinbox(self.frame, from_=min_val, to=max_val, increment=step,
                                      textvariable=self.var, width=10,
                                      validate='key', validatecommand=vcmd)
        
        self.spinbox.pack(side=tk.RIGHT)
        self.spinbox.bind('<Return>', self.on_spinbox_enter)
        self.spinbox.bind('<FocusOut>', self.on_spinbox_enter)
        self.spinbox.bind('<KeyRelease>', self.on_spinbox_key_release)

    def validate_int_input(self, P):
        """整数入力のみを許可するバリデーション関数 ('%P' は変更後のテキスト全体)"""
        if P == "" or P == "-":
            return True  # 入力途中（空やマイナス記号のみ）を許可
        try:
            int(P)
            return True
        except ValueError:
            return False # 整数に変換できない入力は拒否
    
    def on_slider_changed(self, value):
        """スライダー値変更"""
        if self.updating:
            return
        self.updating = True
        try:
            val = float(value)
            stepped_val = round(val / self.step) * self.step
            
            if self.dtype is float:
                stepped_val = round(stepped_val, self.decimals)
            else: # int
                stepped_val = int(stepped_val)
            
            self.var.set(stepped_val)
            if self.callback:
                self.callback(stepped_val)
        finally:
            self.updating = False
    
    def on_spinbox_enter(self, event):
        """スピンボックスでEnter押下時やフォーカスアウト時に値を正規化"""
        if self.updating:
            return
        self.updating = True
        try:
            value_str = self.spinbox.get()
            
            if value_str == "" or value_str == "-":
                val = 0
            else:
                try:
                    val = self.dtype(value_str)
                except ValueError:
                    self.var.set(self.var.get())
                    return

            val = max(self.min_val, min(self.max_val, val))
            
            stepped_val = round(val / self.step) * self.step
            if self.dtype is float:
                stepped_val = round(stepped_val, self.decimals)
            else: # int
                stepped_val = int(stepped_val)
            
            self.var.set(stepped_val)
            if self.callback:
                self.callback(stepped_val)
        finally:
            self.updating = False
    
    def on_spinbox_key_release(self, event):
        """スピンボックスキー離し（リアルタイム更新用）"""
        if event.keysym in ('Up', 'Down'):
            self.on_spinbox_enter(event)
    
    def set_value(self, value):
        """値を設定"""
        if self.updating:
            return
        self.updating = True
        try:
            self.var.set(value)
        finally:
            self.updating = False
    
    def get_value(self):
        """現在の値を取得して、指定されたdtypeに変換して返す"""
        return self.dtype(self.var.get())
    
    def pack(self, **kwargs):
        """フレームをpack"""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """フレームをgrid"""
        self.frame.grid(**kwargs)

class ImageDisplayWidget:
    """画像表示ウィジェット"""
    
    def __init__(self, parent, main_window):
        self.parent = parent
        self.main_window = main_window
        
        self.canvas = Canvas(parent, bg='gray', relief=tk.SUNKEN, bd=2)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        self.canvas.bind('<Configure>', self.on_canvas_resize)
        
        # 初期表示テキスト
        self.display_text = self.main_window.tr("「ファイル」→「写真を開く」で写真を開いて編集しましょう！")
        self.canvas.create_text(400, 300, text=self.display_text, fill='white', 
                               font=('Arial', 12), anchor='center', tags='text')
        
        self.original_image = None
        self.ai_mask_mode = False
        self.full_res_width = 1
        self.full_res_height = 1
        self.current_photo_id = None
        self.current_image_array = None
        self.mouse_pressed = False
    
    def on_click(self, event):
        """マウスクリック処理"""
        self.mouse_pressed = True
        
        if self.main_window.medium_editor:
            if self.ai_mask_mode:
                

                # クリック位置を計算してAIマスク作成
                if self.current_photo_id:
                    bbox = self.canvas.bbox(self.current_photo_id)
                    if bbox:
                        img_x1, img_y1, img_x2, img_y2 = bbox
                        img_width = img_x2 - img_x1
                        img_height = img_y2 - img_y1
                        
                        if img_x1 <= event.x <= img_x2 and img_y1 <= event.y <= img_y2:
                            relative_x = (event.x - img_x1) / img_width
                            relative_y = (event.y - img_y1) / img_height
                            
                            full_res_x = int(relative_x * self.full_res_width)
                            full_res_y = int(relative_y * self.full_res_height)
                            
                            self.main_window.on_image_clicked(full_res_x, full_res_y)
            else:
                # マスク作成モードでない場合、初期画像を表示
                self.set_image(self.main_window.medium_editor.initial_image_array)

    def on_release(self, event):
        """マウスリリース処理"""
        if self.mouse_pressed and not self.ai_mask_mode and self.main_window.medium_editor:
            # マスク作成モードでない場合、通常の表示に戻す
            self.main_window.update_image_display(force_full=True)
        self.mouse_pressed = False
    
    def on_canvas_resize(self, event):
        """キャンバスリサイズ処理"""
        if self.current_image_array is not None:
            self.set_image(self.current_image_array)

    def set_image(self, image_array, is_mask_overlay=False):
        """画像を設定"""
        if image_array is None:
            return
        
        if not is_mask_overlay:
            self.current_image_array = image_array
        
        # numpy配列をPIL Imageに変換
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # PIL ImageからPhotoImageに変換
        pil_image = Image.fromarray(image_array)
        
        # キャンバスサイズに合わせてリサイズ（アスペクト比維持）
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # キャンバスが初期化済み
            # アスペクト比を保ってリサイズ
            img_width, img_height = pil_image.size
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # キャンバスをクリア
        self.canvas.delete('all')
        
        # 画像を中央に配置
        canvas_center_x = canvas_width // 2 if canvas_width > 1 else 400
        canvas_center_y = canvas_height // 2 if canvas_height > 1 else 300
        
        self.current_photo_id = self.canvas.create_image(canvas_center_x, canvas_center_y, 
                                                        image=self.photo, anchor='center')
        
        if not is_mask_overlay:
            self.original_image = pil_image
    
    def set_ai_mask_mode(self, enabled):
        """AIマスクモードの設定"""
        self.ai_mask_mode = enabled
        if enabled:
            self.canvas.config(cursor='crosshair')
        else:
            self.canvas.config(cursor='arrow')
    
    def show_text(self, text):
        """テキストを表示"""
        self.canvas.delete('all')
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        center_x = canvas_width // 2 if canvas_width > 1 else 400
        center_y = canvas_height // 2 if canvas_height > 1 else 300
        self.canvas.create_text(center_x, center_y, text=text, fill='black', 
                               font=('Arial', 12), anchor='center')

class ProgressDialog:
    """プログレスダイアログ"""
    
    def __init__(self, parent, title, message):
        self.dialog = Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("300x100")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        
        
        # 中央に配置
        self.dialog.geometry(f"+{parent.winfo_rootx() + 50}+{parent.winfo_rooty() + 50}")
        
        Label(self.dialog, text=message).pack(pady=20)
        
        # プログレスバー
        self.progress = ttk.Progressbar(self.dialog, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=20, pady=10)
        self.progress.start()
    
    def close(self):
        """ダイアログを閉じる"""
        if self.dialog.winfo_exists():
            self.progress.stop()
            self.dialog.destroy()

class ExportDialog:
    """エクスポート設定ダイアログ"""
    
    def __init__(self, parent, main_window):
        self.parent = parent
        self.main_window = main_window
        self.result = None
        
        self.dialog = Toplevel(parent)
        self.dialog.title(main_window.tr("エクスポート設定"))
        self.dialog.geometry("300x150")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        
        
        # 中央に配置
        self.dialog.geometry(f"+{parent.winfo_rootx() + 100}+{parent.winfo_rooty() + 100}")
        
        # 画質設定
        quality_frame = Frame(self.dialog)
        quality_frame.pack(fill=tk.X, padx=10, pady=5)
        Label(quality_frame, text=main_window.tr("画質:")).pack(side=tk.LEFT)
        
        self.quality_var = StringVar(value="90")
        quality_combo = ttk.Combobox(quality_frame, textvariable=self.quality_var, 
                                    values=[str(q) for q in range(30, 101, 5)], 
                                    state="readonly", width=10)
        quality_combo.pack(side=tk.RIGHT)
        
        # 形式設定
        format_frame = Frame(self.dialog)
        format_frame.pack(fill=tk.X, padx=10, pady=5)
        Label(format_frame, text=main_window.tr("形式:")).pack(side=tk.LEFT)
        
        self.format_var = StringVar(value="JPEG")
        format_combo = ttk.Combobox(format_frame, textvariable=self.format_var, 
                                   values=["JPEG", "PNG"], state="readonly", width=10)
        format_combo.pack(side=tk.RIGHT)
        
        # ボタン
        button_frame = Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        Button(button_frame, text=main_window.tr("キャンセル"), 
               command=self.cancel).pack(side=tk.LEFT, padx=5)
        Button(button_frame, text=main_window.tr("エクスポート"), 
               command=self.accept).pack(side=tk.RIGHT, padx=5)
    
    def accept(self):
        """OK処理"""
        self.result = True
        self.dialog.destroy()
    
    def cancel(self):
        """キャンセル処理"""
        self.result = False
        self.dialog.destroy()
    
    def show(self):
        """ダイアログ表示"""
        self.dialog.wait_window()
        return self.result
    
    def get_quality(self):
        return int(self.quality_var.get())
    
    def get_format(self):
        return self.format_var.get()

class SettingsDialog:
    """設定ダイアログ"""
    
    def __init__(self, parent, settings_manager, main_window):
        self.parent = parent
        self.settings_manager = settings_manager
        self.main_window = main_window
        
        self.dialog = Toplevel(parent)
        self.dialog.title(main_window.tr("設定"))
        self.dialog.geometry("700x550")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        
        # 中央に配置
        self.dialog.geometry(f"+{parent.winfo_rootx() + 100}+{parent.winfo_rooty() + 100}")
        
        # ノートブック（タブ）
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 処理デバイスタブ
        device_frame = Frame(notebook)
        notebook.add(device_frame, text=main_window.tr("処理デバイス"))
        
        Label(device_frame, text=main_window.tr("処理デバイス:")).pack(anchor='w', padx=10, pady=10)
        
        self.device_var = StringVar()
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, 
                                        state="readonly", width=50)
        
        device_options = []
        self.device_indices = []
        
        # Slangpyデバイス一覧を取得
        if RAW_EDITOR_AVAILABLE and raw_image_editor.SLANGPY_AVAILABLE:
            try:
                adapters = raw_image_editor.get_slangpy_adapters()
                if not adapters:
                    device_options.append(main_window.tr("Slangpyデバイスが見つかりません"))
                    self.device_indices.append(0)
                else:
                    for i, adapter_name in enumerate(adapters):
                        device_options.append(adapter_name)
                        self.device_indices.append(i)
            except Exception:
                traceback.print_exc()
                device_options.append(main_window.tr("Slangpyデバイスの取得中にエラー"))
                self.device_indices.append(0)
        else:
            device_options.append(main_window.tr("raw_image_editorが利用できません"))
            self.device_indices.append(0)
        
        self.device_combo['values'] = device_options
        self.device_combo.pack(anchor='w', padx=10, pady=5)
        
        # 言語タブ
        language_frame = Frame(notebook)
        notebook.add(language_frame, text=main_window.tr("言語"))
        
        Label(language_frame, text=main_window.tr("言語:")).pack(anchor='w', padx=10, pady=50)
        
        self.language_var = StringVar()
        language_combo = ttk.Combobox(language_frame, textvariable=self.language_var, 
                                     values=["English", "日本語"], state="readonly", width=50)
        language_combo.pack(anchor='w', padx=10, pady=5)

        # 画像設定タブ
        image_frame = Frame(notebook)
        notebook.add(image_frame, text=main_window.tr("画像"))

        Label(image_frame, text=main_window.tr("画像設定")).pack(anchor='w', padx=10, pady=10)

        # UIプレビューサイズ
        Label(image_frame, text=main_window.tr("UIプレビューサイズ(長辺):")).pack(anchor='w', padx=10, pady=(10, 0))
        self.preview_size_slider = SliderWithSpinBox(image_frame, 500, 2000, 50, dtype=int, initial_value=1000)
        self.preview_size_slider.pack(fill=tk.X, padx=10, pady=2)

        # ドラッグ中プレビューサイズ
        Label(image_frame, text=main_window.tr("ドラッグ中プレビューサイズ(長辺):")).pack(anchor='w', padx=10, pady=(10, 0))
        self.dragging_preview_size_slider = SliderWithSpinBox(image_frame, 100, 2000, 50, dtype=int, initial_value=300)
        self.dragging_preview_size_slider.pack(fill=tk.X, padx=10, pady=2)
        
        # 保存ボタン
        Button(self.dialog, text=main_window.tr("設定を保存"), 
               command=self.save_settings).pack(pady=10)
        
        # 設定を読み込み
        self.load_settings()
    
    def load_settings(self):
        """設定を読み込みUIに反映"""
        device_idx = self.settings_manager.device_index
        try:
            # 保存されているデバイスインデックスが、利用可能なデバイスインデックスリストのどこにあるか探す
            combo_box_index = self.device_indices.index(device_idx)
            self.device_var.set(self.device_combo['values'][combo_box_index])
        except ValueError:
            
            # 保存されたインデックスが見つからない場合（例：GPUが取り外された）、最初の選択肢にフォールバック
            if self.device_combo['values']:
                self.device_var.set(self.device_combo['values'][0])
        
        self.language_var.set(self.settings_manager.language)
        self.preview_size_slider.set_value(self.settings_manager.preview_size)
        self.dragging_preview_size_slider.set_value(self.settings_manager.dragging_preview_size)
    
    def save_settings(self):
        """UIから設定を取得し保存"""
        # 画像設定のバリデーション
        preview_size = self.preview_size_slider.get_value()
        dragging_preview_size = self.dragging_preview_size_slider.get_value()
        if dragging_preview_size >= preview_size:
            messagebox.showerror(self.main_window.tr("警告"), 
                               self.main_window.tr("ドラッグ中のプレビューサイズは、UIプレビューサイズより小さくしてください。"))
            return

        try:
            device_text = self.device_var.get()
            combo_box_index = list(self.device_combo['values']).index(device_text)
            device_idx = self.device_indices[combo_box_index]
        except (ValueError, IndexError):

            # 選択が無効な場合は最初のデバイスにフォールバック
            device_idx = self.device_indices[0] if self.device_indices else 0

        self.settings_manager.device_index = device_idx
        self.settings_manager.language = self.language_var.get()
        self.settings_manager.preview_size = preview_size
        self.settings_manager.dragging_preview_size = dragging_preview_size
        
        try:
            self.settings_manager.save()
            messagebox.showinfo(self.main_window.tr("情報"), 
                              self.main_window.tr("設定を保存しました。ソフトウェアを再起動してください。"))
            self.dialog.destroy()
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror(self.main_window.tr("警告"), 
                               f"{self.main_window.tr('設定の保存に失敗しました')}: {str(e)}")

class RAWDevelopmentGUI:
    """RAW現像ソフトのメインウィンドウ"""
    
    def __init__(self):
        # 設定マネージャーを初期化し、設定を読み込む
        self.settings_manager = SettingsManager()
        self.settings_manager.__post_init__()
        self.settings_manager.load()

        # 翻訳を読み込む
        self.translations = {}
        self.load_translations()

        # Slangpyが利用可能かチェック
        if not (RAW_EDITOR_AVAILABLE and raw_image_editor.SLANGPY_AVAILABLE):
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                self.tr("初期化エラー"),
                self.tr("画像処理エンジンの初期化に失敗しました。")
            )
            sys.exit(1)

        # 初期化
        self.editor = None
        self.medium_editor = None
        self.small_editor = None
        self.current_file_path = None
        self.device_index = self.settings_manager.device_index
        self.dragging = False
        self.mask_display_enabled = False
        self.drag_timer_id = None

        # raw_image_editorの初期化
        if RAW_EDITOR_AVAILABLE:
            try:
                raw_image_editor.init(self.device_index)
                
                raw_image_editor.photo_metadata.set_exiftool_path(self.settings_manager.exiftool_path)
                
            except Exception as e:
                print(f"raw_image_editor の初期化に失敗: {e}")
                traceback.print_exc()
        
        # 編集パラメータ（マスクごと）
        self.mask_edit_params: Dict[str, EditParameters] = {self.tr("マスクなし"): EditParameters()}
        self.current_mask_name = self.tr("マスクなし")
        
        # UI初期化
        self.init_ui()
    
    def tr(self, key: str) -> str:
        """指定されたキーの翻訳済みテキストを返す"""
        return self.translations.get(self.settings_manager.language, {}).get(key, key)

    def load_translations(self):
        """言語データをコード内から直接読み込む"""
        
        translations_data = {
            "English": {
                "RawPhotoForge": "RawPhotoForge",
                "Slangpyデバイスが見つかりません": "No Slangpy devices found",
                "Slangpyデバイスの取得中にエラー": "Error while getting Slangpy devices",
                "エクスポートに失敗しました": "Export failed",
                "AIマスクの作成に失敗しました": "Failed to create AI mask",
                "画像の読み込みに失敗しました": "Failed to load image",

                "RAW現像ソフト": "RAW Development Software",
                "ファイル": "File",
                "写真を開く": "Open Photo",
                "画像をエクスポート": "Export Image",
                "写真をエクスポート": "Export Image",
                "編集": "Edit",
                "すべての編集をリセット": "Reset All Edits",
                "プリセット": "Preset",
                "現在の編集をプリセットとして保存": "Save Current Edits as Preset",
                "プリセットを読み込み": "Load Preset",
                "設定": "Settings",
                "露出:": "Exposure",
                "露出": "Exposure",
                "基本的な露出調整を行います": "Adjust basic exposure.",
                "コントラスト:": "Contrast:",
                "コントラスト": "Contrast",
                "シャドウ:": "Shadow:",
                "シャドウ": "Shadow",
                "ハイライト:": "Highlight:",
                "ハイライト": "Highlight",
                "黒レベル:": "Black Level:",
                "黒レベル": "Black Level",
                "白レベル:": "White Level:",
                "白レベル": "White Level",
                "このタブをリセット": "Reset This Tab",
                "明るさカーブ": "Brightness Curve",
                "明るさのトーンカーブを調整します": "Adjust the brightness tone curve.",
                "トーンカーブをリセット": "Reset Tone Curve",
                "OKLCH Hカーブ": "OKLCH H Curve",
                "OKLCHの色相(H)を調整します": "Adjust the hue (H) of OKLCH.",
                "OKLCH Cカーブ": "OKLCH C Curve",
                "OKLCHの彩度(C)を調整します": "Adjust the chroma (C) of OKLCH.",
                "OKLCH Lカーブ": "OKLCH L Curve",
                "OKLCHの輝度(L)を調整します": "Adjust the lightness (L) of OKLCH.",
                "WB": "WB",
                "効果": "Effects",
                "周辺減光": "Vignette",
                "周辺減光を調整します": "Adjust vignette effect.",
                "周辺減光:": "Vignette:",
                "ホワイトバランスを調整します": "Adjust white balance.",
                "色温度:": "Temperature:",
                "色温度": "Temperature",
                "色かぶり補正:": "Tint:",
                "色かぶり補正": "Tint",
                "マスク": "Mask",
                "AIマスク作成": "Create AI Mask",
                "キャンセル": "Cancel",
                "マスクを選択": "Select Mask",
                "マスクを削除": "Delete Mask",
                "マスクなし": "No Mask",
                "マスクを表示": "Show Mask",
                "マスクを反転して新規作成": "Invert Mask and Create New",
                "マスク範囲の微調整:": "Fine-tune Mask Range:",
                "マスク範囲の微調整": "Fine-tune Mask Range",
                "メタデータ": "Metadata",
                "画像のメタデータ情報": "Image metadata information.",
                "キー": "Key",
                "値": "Value",
                "処理デバイス": "Processing Device",
                "言語": "Language",
                "言語:": "Language",
                "処理デバイス:": "Processing Device:",
                "設定を保存": "Save Settings",
                "エクスポート設定": "Export Settings",
                "画質:": "Quality:",
                "形式:": "Format:",
                "エクスポート": "Export",
                "情報": "Information",
                "設定を保存しました。ソフトウェアを再起動してください。": "Settings saved. Please restart the software.",
                "警告": "Warning",
                "設定の保存に失敗しました": "Failed to save settings",
                "画像ファイル": "Image Files",
                "すべてのファイル": "All Files",
                "エラー": "Error",
                "raw_image_editor が利用できません": "raw_image_editor is not available",
                "読み込み中": "Loading",
                "画像を読み込み中...": "Loading image...",
                "エクスポートする画像がありません": "No image to export",
                "エクスポート中": "Exporting",
                "画像をエクスポート中...": "Exporting image...",
                "エクスポートの準備中にエラーが発生しました": "Error during export preparation",
                "エクスポート完了": "Export Complete",
                "画像を保存しました": "Image saved",
                "プリセットを保存": "Save Preset",
                "プリセットファイル": "Preset Files",
                "プリセットを保存しました": "Preset saved",
                "プリセットの保存に失敗しました": "Failed to save preset",
                "プリセットを読み込みました": "Preset loaded",
                "プリセットの読み込みに失敗しました": "Failed to load preset",
                "画像が読み込まれていません": "No image loaded",
                "AIマスク作成中": "Creating AI Mask",
                "AIマスク作成中...": "Creating AI mask...",
                "AIマスクを作成しました": "AI mask created",
                "AIマスク作成完了": "AI Mask Creation Complete",
                "AIマスクの準備中にエラーが発生しました": "Error during AI mask preparation",
                "確認": "Confirm",
                "を削除しますか？": " will be deleted. Are you sure?",
                "反転するマスクが選択されていません": "No mask selected to invert",
                "反転マスクを作成しました": "Inverted mask created",
                "マスク作成完了": "Mask Creation Complete",
                "初期化エラー": "Initialization Error",
                "画像処理エンジンの初期化に失敗しました。": "Failed to initialize image processing engine.",
                "「ファイル」→「写真を開く」で写真を開いて編集しましょう！": "'File' -> 'Open Photo' to open a photo and start editing!",
                "画像": "Image",
                "画像設定": "Image Settings",
                "UIプレビューサイズ(長辺):": "UI Preview Size (long edge):",
                "ドラッグ中プレビューサイズ(長辺):": "Dragging Preview Size (long edge):",
                "ドラッグ中のプレビューサイズは、UIプレビューサイズより小さくしてください。": "Dragging preview size must be smaller than UI preview size."
            },
            "日本語": {
                "RawPhotoForge": "RawPhotoForge",
                "Slangpyデバイスが見つかりません": "Slangpyデバイスが見つかりません",
                "Slangpyデバイスの取得中にエラー": "Slangpyデバイスの取得中にエラー",
                "エクスポートに失敗しました": "エクスポートに失敗しました",
                "AIマスクの作成に失敗しました": "AIマスクの作成に失敗しました",
                "画像の読み込みに失敗しました": "画像の読み込みに失敗しました",
                
                "RAW現像ソフト": "RAW現像ソフト",
                "ファイル": "ファイル",
                "写真を開く": "写真を開く",
                "画像をエクスポート": "画像をエクスポート",
                "写真をエクスポート": "写真をエクスポート",
                "編集": "編集",
                "すべての編集をリセット": "すべての編集をリセット",
                "プリセット": "プリセット",
                "現在の編集をプリセットとして保存": "現在の編集をプリセットとして保存",
                "プリセットを読み込み": "プリセットを読み込み",
                "設定": "設定",
                "露出": "露出",
                "基本的な露出調整を行います": "基本的な露出調整を行います",
                "コントラスト": "コントラスト",
                "シャドウ": "シャドウ",
                "ハイライト": "ハイライト",
                "黒レベル": "黒レベル",
                "白レベル": "白レベル",
                "このタブをリセット": "このタブをリセット",
                "明るさカーブ": "明るさカーブ",
                "明るさのトーンカーブを調整します": "明るさのトーンカーブを調整します",
                "トーンカーブをリセット": "トーンカーブをリセット",
                "OKLCH Hカーブ": "OKLCH Hカーブ",
                "OKLCHの色相(H)を調整します": "OKLCHの色相(H)を調整します",
                "OKLCH Cカーブ": "OKLCH Cカーブ",
                "OKLCHの彩度(C)を調整します": "OKLCHの彩度(C)を調整します",
                "OKLCH Lカーブ": "OKLCH Lカーブ",
                "OKLCHの輝度(L)を調整します": "OKLCHの輝度(L)を調整します",
                "WB": "WB",
                "効果": "効果",
                "周辺減光": "周辺減光",
                "周辺減光を調整します": "周辺減光を調整します",
                "周辺減光:": "周辺減光:",
                "ホワイトバランスを調整します": "ホワイトバランスを調整します",
                "色温度": "色温度",
                "色かぶり補正": "色かぶり補正",
                "マスク": "マスク",
                "AIマスク作成": "AIマスク作成",
                "キャンセル": "キャンセル",
                "マスクを選択": "マスクを選択",
                "マスクを削除": "マスクを削除",
                "マスクなし": "マスクなし",
                "マスクを表示": "マスクを表示",
                "マスクを反転して新規作成": "マスクを反転して新規作成",
                "マスク範囲の微調整": "マスク範囲の微調整",
                "メタデータ": "メタデータ",
                "画像のメタデータ情報": "画像のメタデータ情報",
                "キー": "キー",
                "値": "値",
                "処理デバイス": "処理デバイス",
                "言語": "言語",
                "設定を保存": "設定を保存",
                "エクスポート設定": "エクスポート設定",
                "画質:": "画質:",
                "形式:": "形式:",
                "エクスポート": "エクスポート",
                "情報": "情報",
                "設定を保存しました。ソフトウェアを再起動してください。": "設定を保存しました。ソフトウェアを再起動してください。",
                "警告": "警告",
                "設定の保存に失敗しました": "設定の保存に失敗しました",
                "画像ファイル": "画像ファイル",
                "すべてのファイル": "すべてのファイル",
                "エラー": "エラー",
                "raw_image_editor が利用できません": "raw_image_editor が利用できません",
                "読み込み中": "読み込み中",
                "画像を読み込み中...": "画像を読み込み中...",
                "エクスポートする画像がありません": "エクスポートする画像がありません",
                "エクスポート中": "エクスポート中",
                "画像をエクスポート中...": "画像をエクスポート中...",
                "エクスポートの準備中にエラーが発生しました": "エクスポートの準備中にエラーが発生しました",
                "エクスポート完了": "エクスポート完了",
                "画像を保存しました": "画像を保存しました",
                "プリセットを保存": "プリセットを保存",
                "プリセットファイル": "プリセットファイル",
                "プリセットを保存しました": "プリセットを保存しました",
                "プリセットの保存に失敗しました": "プリセットの保存に失敗しました",
                "プリセットを読み込みました": "プリセットを読み込みました",
                "プリセットの読み込みに失敗しました": "プリセットの読み込みに失敗しました",
                "画像が読み込まれていません": "画像が読み込まれていません",
                "AIマスク作成中": "AIマスク作成中",
                "AIマスク作成中...": "AIマスク作成中...",
                "AIマスクを作成しました": "AIマスクを作成しました",
                "AIマスク作成完了": "AIマスク作成完了",
                "AIマスクの準備中にエラーが発生しました": "AIマスクの準備中にエラーが発生しました",
                "確認": "確認",
                "を削除しますか？": "を削除しますか？",
                "反転するマスクが選択されていません": "反転するマスクが選択されていません",
                "反転マスクを作成しました": "反転マスクを作成しました",
                "マスク作成完了": "マスク作成完了",
                "初期化エラー": "初期化エラー",
                "画像処理エンジンの初期化に失敗しました。": "画像処理エンジンの初期化に失敗しました。",
                "「ファイル」→「写真を開く」で写真を開いて編集しましょう！": "「ファイル」→「写真を開く」で写真を開いて編集しましょう！",
                "画像": "画像",
                "画像設定": "画像設定",
                "UIプレビューサイズ(長辺):": "UIプレビューサイズ(長辺):",
                "ドラッグ中プレビューサイズ(長辺):": "ドラッグ中プレビューサイズ(長辺):",
                "ドラッグ中のプレビューサイズは、UIプレビューサイズより小さくしてください。": "ドラッグ中のプレビューサイズは、UIプレビューサイズより小さくしてください。"
            }
        }
        self.translations = translations_data

    @property
    def edit_params(self) -> EditParameters:
        """現在のマスクに対応する編集パラメータを取得"""
        return self.mask_edit_params.setdefault(self.current_mask_name, EditParameters())

    def init_ui(self):
        """UI初期化"""
        self.root = tk.Tk()
        self.root.title("RawPhotoForge")
        self.root.geometry("1000x600")
        
        # メニューバー
        self.create_menu_bar()
        
        # メインフレーム
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # PanedWindow（左右分割）
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # 左側：画像表示（6の割合）
        left_frame = Frame(paned_window)
        paned_window.add(left_frame, weight=6)
        
        self.image_widget = ImageDisplayWidget(left_frame, self)
        
        # 右側：編集パネル（4の割合）
        right_frame = Frame(paned_window)
        paned_window.add(right_frame, weight=4)
        
        self.edit_panel = self.create_edit_panel(right_frame)

        # サッシュの初期位置を6:4に設定 (一度だけ実行)
        def configure_sash(event):
            paned_window.sashpos(0, int(event.width * 0.6))
            paned_window.unbind('<Configure>')
        paned_window.bind('<Configure>', configure_sash)
    
    def create_menu_bar(self):
        """メニューバー作成"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # ファイルメニュー
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.tr("ファイル"), menu=file_menu)
        
        file_menu.add_command(label=self.tr("写真を開く"), command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label=self.tr("写真をエクスポート"), command=self.export_image, accelerator="Ctrl+E")
        
        # 編集メニュー
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.tr("編集"), menu=edit_menu)
        
        edit_menu.add_command(label=self.tr("すべての編集をリセット"), command=self.reset_all_edits)
        
        # プリセットメニュー
        preset_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.tr("プリセット"), menu=preset_menu)
        
        preset_menu.add_command(label=self.tr("現在の編集をプリセットとして保存"), command=self.save_preset)
        preset_menu.add_command(label=self.tr("プリセットを読み込み"), command=self.load_preset)
        
        # 設定メニュー
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.tr("設定"), menu=settings_menu)
        
        settings_menu.add_command(label=self.tr("設定"), command=self.show_settings)
        
        # キーボードショートカット
        self.root.bind('<Control-o>', lambda e: self.open_file())
        self.root.bind('<Control-e>', lambda e: self.export_image())
    
    def create_edit_panel(self, parent):
        """編集パネル作成"""
        # タブウィジェット
        self.tab_widget = ttk.Notebook(parent)
        self.tab_widget.pack(fill=tk.BOTH, expand=True)
        
        # 各タブを作成
        self.create_exposure_tab()
        self.create_tone_curve_tabs()
        self.create_wb_tab()
        self.create_effects_tab()
        self.create_mask_tab()
        self.create_metadata_tab()
        
        return self.tab_widget
    
    def create_exposure_tab(self):
        """露出タブ作成"""
        tab = Frame(self.tab_widget)
        self.tab_widget.add(tab, text=self.tr("露出"))
        
        # スクロール可能フレーム
        canvas = Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # キャンバスのサイズ変更に追従して、中のフレームの幅を更新する
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(canvas_window, width=e.width))
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 説明
        Label(scrollable_frame, text=self.tr("基本的な露出調整を行います")).pack(pady=5)
        
        # 露出
        Label(scrollable_frame, text=self.tr("露出:")).pack(anchor='w', padx=10, pady=(10, 0))
        self.exposure_slider = SliderWithSpinBox(scrollable_frame, -6.0, 6.0, 0.05, dtype=float, decimals=2, initial_value=0.0, 
                                                callback=self.on_exposure_changed)
        self.exposure_slider.pack(fill=tk.X, padx=10, pady=2)
        
        # コントラスト
        Label(scrollable_frame, text=self.tr("コントラスト:")).pack(anchor='w', padx=10, pady=(10, 0))
        self.contrast_slider = SliderWithSpinBox(scrollable_frame, -100, 100, 1, dtype=int, initial_value=0, 
                                                callback=self.on_contrast_changed)
        self.contrast_slider.pack(fill=tk.X, padx=10, pady=2)
        
        # シャドウ
        Label(scrollable_frame, text=self.tr("シャドウ:")).pack(anchor='w', padx=10, pady=(10, 0))
        self.shadow_slider = SliderWithSpinBox(scrollable_frame, -100, 100, 1, dtype=int, initial_value=0, 
                                              callback=self.on_shadow_changed)
        self.shadow_slider.pack(fill=tk.X, padx=10, pady=2)
        
        # ハイライト
        Label(scrollable_frame, text=self.tr("ハイライト:")).pack(anchor='w', padx=10, pady=(10, 0))
        self.highlight_slider = SliderWithSpinBox(scrollable_frame, -100, 100, 1, dtype=int, initial_value=0, 
                                                 callback=self.on_highlight_changed)
        self.highlight_slider.pack(fill=tk.X, padx=10, pady=2)
        
        # 黒レベル
        Label(scrollable_frame, text=self.tr("黒レベル:")).pack(anchor='w', padx=10, pady=(10, 0))
        self.black_slider = SliderWithSpinBox(scrollable_frame, -100, 100, 1, dtype=int, initial_value=0, 
                                             callback=self.on_black_changed)
        self.black_slider.pack(fill=tk.X, padx=10, pady=2)
        
        # 白レベル
        Label(scrollable_frame, text=self.tr("白レベル:")).pack(anchor='w', padx=10, pady=(10, 0))
        self.white_slider = SliderWithSpinBox(scrollable_frame, -100, 100, 1, dtype=int, initial_value=0, 
                                             callback=self.on_white_changed)
        self.white_slider.pack(fill=tk.X, padx=10, pady=2)
        
        # リセットボタン
        Button(scrollable_frame, text=self.tr("このタブをリセット"), 
               command=self.reset_exposure_tab).pack(pady=20)
    
    def create_tone_curve_tabs(self):
        """トーンカーブタブ群作成"""
        self.create_brightness_curve_tab()
        self.create_oklch_h_curve_tab()
        self.create_oklch_c_curve_tab()
        self.create_oklch_l_curve_tab()
    
    def create_brightness_curve_tab(self):
        """明るさカーブタブ作成"""
        tab = Frame(self.tab_widget)
        self.tab_widget.add(tab, text=self.tr("明るさカーブ"))
        
        Label(tab, text=self.tr("明るさのトーンカーブを調整します")).pack(pady=5)
        
        curve_frame = Frame(tab)
        curve_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.brightness_curve = ToneCurveWidget(curve_frame, "brightness", self)
        
        Button(tab, text=self.tr("トーンカーブをリセット"), 
               command=self.brightness_curve.reset_curve).pack(pady=5)
    
    def create_oklch_h_curve_tab(self):
        """OKLCH Hカーブタブ作成"""
        tab = Frame(self.tab_widget)
        self.tab_widget.add(tab, text=self.tr("OKLCH Hカーブ"))
        
        Label(tab, text=self.tr("OKLCHの色相(H)を調整します")).pack(pady=5)
        
        curve_frame = Frame(tab)
        curve_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.oklch_h_curve = ToneCurveWidget(curve_frame, "oklch_h", self)
        
        Button(tab, text=self.tr("トーンカーブをリセット"), 
               command=self.oklch_h_curve.reset_curve).pack(pady=5)
    
    def create_oklch_c_curve_tab(self):
        """OKLCH Cカーブタブ作成"""
        tab = Frame(self.tab_widget)
        self.tab_widget.add(tab, text=self.tr("OKLCH Cカーブ"))
        
        Label(tab, text=self.tr("OKLCHの彩度(C)を調整します")).pack(pady=5)
        
        curve_frame = Frame(tab)
        curve_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.oklch_c_curve = ToneCurveWidget(curve_frame, "oklch_c", self)
        
        Button(tab, text=self.tr("トーンカーブをリセット"), 
               command=self.oklch_c_curve.reset_curve).pack(pady=5)
    
    def create_oklch_l_curve_tab(self):
        """OKLCH Lカーブタブ作成"""
        tab = Frame(self.tab_widget)
        self.tab_widget.add(tab, text=self.tr("OKLCH Lカーブ"))
        
        Label(tab, text=self.tr("OKLCHの輝度(L)を調整します")).pack(pady=5)
        
        curve_frame = Frame(tab)
        curve_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.oklch_l_curve = ToneCurveWidget(curve_frame, "oklch_l", self)
        
        Button(tab, text=self.tr("トーンカーブをリセット"), 
               command=self.oklch_l_curve.reset_curve).pack(pady=5)
    
    def create_wb_tab(self):
        """ホワイトバランスタブ作成"""
        tab = Frame(self.tab_widget)
        self.tab_widget.add(tab, text=self.tr("WB"))
        
        # スクロール可能フレーム
        canvas = Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # キャンバスのサイズ変更に追従して、中のフレームの幅を更新する
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(canvas_window, width=e.width))
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        Label(scrollable_frame, text=self.tr("ホワイトバランスを調整します")).pack(pady=5)
        
        # 色温度
        Label(scrollable_frame, text=self.tr("色温度:")).pack(anchor='w', padx=10, pady=(10, 0))
        self.wb_temperature_slider = SliderWithSpinBox(scrollable_frame, -100, 100, 1, dtype=int, initial_value=0, 
                                                      callback=self.on_wb_temperature_changed)
        self.wb_temperature_slider.pack(fill=tk.X, padx=10, pady=2)
        
        # 色かぶり補正
        Label(scrollable_frame, text=self.tr("色かぶり補正:")).pack(anchor='w', padx=10, pady=(10, 0))
        self.wb_tint_slider = SliderWithSpinBox(scrollable_frame, -100, 100, 1, dtype=int, initial_value=0, 
                                               callback=self.on_wb_tint_changed)
        self.wb_tint_slider.pack(fill=tk.X, padx=10, pady=2)
        
        # リセットボタン
        Button(scrollable_frame, text=self.tr("このタブをリセット"), 
               command=self.reset_wb_tab).pack(pady=20)

    def create_effects_tab(self):
        """効果タブ作成"""
        tab = Frame(self.tab_widget)
        self.tab_widget.add(tab, text=self.tr("効果"))
        
        canvas = Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.bind('<Configure>', lambda e: canvas.itemconfig(canvas_window, width=e.width))
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        Label(scrollable_frame, text=self.tr("周辺減光を調整します")).pack(pady=5)
        
        Label(scrollable_frame, text=self.tr("周辺減光:")).pack(anchor='w', padx=10, pady=(10, 0))
        self.vignette_slider = SliderWithSpinBox(scrollable_frame, -100, 100, 1, dtype=int, initial_value=0, 
                                                  callback=self.on_vignette_changed)
        self.vignette_slider.pack(fill=tk.X, padx=10, pady=2)
        
        Button(scrollable_frame, text=self.tr("このタブをリセット"), 
               command=self.reset_effects_tab).pack(pady=20)

    
    
    def create_mask_tab(self):
        """マスクタブ作成"""
        tab = Frame(self.tab_widget)
        self.tab_widget.add(tab, text=self.tr("マスク"))
        
        # スクロール可能フレーム
        canvas = Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # キャンバスのサイズ変更に追従して、中のフレームの幅を更新する
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(canvas_window, width=e.width))
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # マスク作成
        create_frame = Frame(scrollable_frame)
        create_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.ai_mask_btn = Button(create_frame, text=self.tr("AIマスク作成"), 
                                 command=self.start_ai_mask_creation)
        self.ai_mask_btn.pack(side=tk.LEFT, padx=5)
        
        self.cancel_mask_btn = Button(create_frame, text=self.tr("キャンセル"), 
                                     command=self.cancel_ai_mask_creation, state=tk.DISABLED)
        self.cancel_mask_btn.pack(side=tk.LEFT, padx=5)
        
        # マスク選択
        select_frame = Frame(scrollable_frame)
        select_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.mask_var = StringVar(value=self.tr("マスクなし"))
        self.mask_combo = ttk.Combobox(select_frame, textvariable=self.mask_var, 
                                      values=[self.tr("マスクなし")], state="readonly")
        self.mask_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.mask_combo.bind('<<ComboboxSelected>>', self.on_mask_selection_changed)
        
        self.delete_mask_btn = Button(select_frame, text=self.tr("マスクを削除"), 
                                     command=self.delete_current_mask, state=tk.DISABLED)
        self.delete_mask_btn.pack(side=tk.RIGHT)
        
        # マスクを表示
        self.mask_display_var = BooleanVar()
        self.mask_display_check = Checkbutton(scrollable_frame, text=self.tr("マスクを表示"), 
                                        variable=self.mask_display_var, 
                                        command=self.on_mask_display_toggled, state=tk.DISABLED)
        self.mask_display_check.pack(anchor='w', padx=10, pady=5)
        
        # マスクを反転して新規作成
        self.invert_mask_btn = Button(scrollable_frame, text=self.tr("マスクを反転して新規作成"), 
                                     command=self.invert_current_mask, state=tk.DISABLED)
        self.invert_mask_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # マスク範囲の微調整
        Label(scrollable_frame, text=self.tr("マスク範囲の微調整:")).pack(anchor='w', padx=10, pady=(10, 0))
        self.mask_range_slider = SliderWithSpinBox(scrollable_frame, -4.0, 4.0, 0.02, dtype=float, decimals=2, initial_value=0.0, 
                                                  callback=self.on_mask_range_changed)
        self.mask_range_slider.slider.config(state='disabled')
        self.mask_range_slider.spinbox.config(state='disabled')
        self.mask_range_slider.pack(fill=tk.X, padx=10, pady=2)
    
    def create_metadata_tab(self):
        """
        メタデータタブを作成する。
        Treeviewの代わりにgridでラベルを配置し、列幅が自動調整されるように変更。
        """
        tab = Frame(self.tab_widget)
        self.tab_widget.add(tab, text=self.tr("メタデータ"))

        Label(tab, text=self.tr("画像のメタデータ情報")).pack(pady=5)

        # スクロール可能なコンテナを作成
        container = Frame(tab)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        canvas = Canvas(container)
        scrollbar_y = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
        
        # データ表示用のフレーム
        self.metadata_frame = Frame(canvas)
        
        self.metadata_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.metadata_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        # --- マウスホイール処理 ---
        def _on_vertical_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_horizontal_mousewheel(event):
            canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_scroll_events(event):
            # 縦スクロール
            canvas.bind_all("<MouseWheel>", _on_vertical_mousewheel)
            # 横スクロール (Shift + ホイール)
            canvas.bind_all("<Shift-MouseWheel>", _on_horizontal_mousewheel)

        def _unbind_scroll_events(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Shift-MouseWheel>")

        # マウスがこのフレーム内に入った時だけ、マウスホイールイベントを有効にする
        self.metadata_frame.bind("<Enter>", _bind_scroll_events)
        self.metadata_frame.bind("<Leave>", _unbind_scroll_events)
        # --- ここまで ---

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        
       
        
    
    def open_file(self):
        """ファイルを開く"""
        file_path = filedialog.askopenfilename(
            title=self.tr("写真を開く"),
            filetypes=[
                (self.tr("画像ファイル"), "*.raw *.cr2 *.cr3 *.nef *.arw *.dng *.jpg *.jpeg *.png *.tiff *.tif"),
                (self.tr("すべてのファイル"), "*.*")
            ]
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """画像を読み込み"""
        if not RAW_EDITOR_AVAILABLE:
            messagebox.showerror(self.tr("エラー"), self.tr("raw_image_editor が利用できません"))
            return
        
        # プログレスダイアログ表示
        progress = ProgressDialog(self.root, self.tr("読み込み中"), self.tr("画像を読み込み中..."))
        
        # ワーカースレッドで読み込み
        def load_worker():
            try:
                # 画像を読み込み
                editor = raw_image_editor.RAWImageEditor(file_path=file_path, apply_clahe=False, lens_correction=True, gamma=(2.222, 4.5))
                
                # デフォルトクロップを適用
                if "EXIF:DefaultCropOrigin" in editor.metadata.metadata:
                    crop_origin = editor.metadata.metadata["EXIF:DefaultCropOrigin"].split(" ")
                    if len(crop_origin) >= 2:
                        crop_x, crop_y = int(crop_origin[0]), int(crop_origin[1])
                        editor.crop(
                            (crop_x, crop_y),
                            (editor.width - crop_x, editor.height - crop_y),
                            update_initial_image_array=True
                        )
                
                # 初期処理を適用
                editor.apply_adjustments()
                
                # リサイズ用のエディターを作成
                medium_editor = self.create_resized_editor(editor, self.settings_manager.preview_size)
                small_editor = self.create_resized_editor(editor, self.settings_manager.dragging_preview_size)
                
                # UIスレッドで結果を処理
                self.root.after(0, lambda: self.on_image_loaded(editor, medium_editor, small_editor, progress))
                
            except Exception as e:
                traceback.print_exc()
                err = str(e)
                # UIスレッドでエラーを処理
                self.root.after(0, lambda: self.on_load_error(err, progress))
                traceback.print_exc()
        
        # 別スレッドで実行
        threading.Thread(target=load_worker, daemon=True).start()
    
    def create_resized_editor(self, original_editor, target_size):
        """リサイズされたエディターを作成"""
        # 元の画像をuint8形式で取得
        original_image = original_editor.as_uint8()
        
        # リサイズ
        h, w = original_image.shape[:2]
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized_image = original_image
        
        # float32に変換（0-1の範囲）
        resized_image = resized_image.astype(np.float32) / 255.0
        
        # 新しいエディターを作成
        resized_editor = raw_image_editor.RAWImageEditor(
            image_array=resized_image,
            metadata=original_editor.metadata
        )
        
        return resized_editor
    
    def on_image_loaded(self, editor, medium_editor, small_editor, progress):
        """画像読み込み完了"""
        progress.close()
        
        self.editor = editor
        self.medium_editor = medium_editor
        self.small_editor = small_editor
        self.current_file_path = editor.file_path

        # フル解像度を記憶
        self.image_widget.full_res_width = editor.width
        self.image_widget.full_res_height = editor.height

        # 編集パラメータをリセット
        self.mask_edit_params = {self.tr("マスクなし"): EditParameters()}
        self.current_mask_name = self.tr("マスクなし")
        
        # UIとマスクコンボボックスをリセット
        self.mask_var.set(self.tr("マスクなし"))
        self.mask_combo['values'] = [self.tr("マスクなし")]

        self.update_ui_from_parameters()
        
        # 画像表示を更新
        self.update_image_display()
        
        # メタデータ表示を更新
        self.update_metadata_display()

        # マスクコントロールの状態を更新
        self._update_mask_controls_state()
    
    def on_load_error(self, error_message, progress):
        """画像読み込みエラー"""
        progress.close()
        messagebox.showerror(self.tr("エラー"), f"{self.tr('画像の読み込みに失敗しました')}: {error_message}")
    
    def update_image_display(self, force_full=False):
        """画像表示を更新"""
        if self.editor is None:
            return
        
        time_start = time.time()
        
        # 表示するエディターを選択
        is_curve_dragging = (hasattr(self, 'brightness_curve') and self.brightness_curve.dragging) or \
                            (hasattr(self, 'oklch_h_curve') and self.oklch_h_curve.dragging) or \
                            (hasattr(self, 'oklch_c_curve') and self.oklch_c_curve.dragging) or \
                            (hasattr(self, 'oklch_l_curve') and self.oklch_l_curve.dragging)

        if (self.dragging or is_curve_dragging) and not force_full:
            display_editor = self.small_editor
        else:
            display_editor = self.medium_editor
        
        try:
            # エディターに編集パラメータを適用
            self.apply_parameters_to_editor(display_editor)
            display_editor.apply_adjustments()
            
            # 画像を取得
            image_array = display_editor.as_uint8()

            # ヒストグラムを計算してトーンカーブウィジェットに渡す
            if image_array is not None and hasattr(self, 'brightness_curve'):
                hist_data = {}
                # グレースケールヒストグラム（白）
                gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                hist_data['white'] = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
                
                # RGBヒストグラム
                colors = ('r', 'g', 'b')
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([image_array], [i], None, [256], [0, 256])
                    hist_data[color] = hist
                
                self.brightness_curve.set_histogram(hist_data)
            
            # マスクオーバーレイ
            if self.mask_display_enabled and self.current_mask_name and self.current_mask_name != self.tr("マスクなし"):
                if self.current_mask_name in display_editor.mask_adjustments_values:
                    mask_data = display_editor.mask_adjustments_values[self.current_mask_name]
                    mask_range_value = self.edit_params.mask_range
                    boolean_mask = mask_range_value < mask_data["ndarray"]
                    mask_image_pil = display_editor.get_mask_image(boolean_mask)
                    
                    # PIL ImageをOpenCV形式に変換
                    overlay_array = np.array(mask_image_pil)
                    if overlay_array.shape[2] == 4:  # RGBA
                        overlay_array = cv2.cvtColor(overlay_array, cv2.COLOR_RGBA2RGB)
                    
                    # オーバーレイが画像と同じサイズかチェック
                    if overlay_array.shape[:2] != image_array.shape[:2]:
                        overlay_array = cv2.resize(overlay_array, (image_array.shape[1], image_array.shape[0]))
                    
                    self.image_widget.set_image(overlay_array, is_mask_overlay=True)
                else:
                    # マスクが見つからない場合は通常の画像を表示
                    self.image_widget.set_image(image_array)
            else:
                # マスク表示が無効な場合は通常の画像を表示
                self.image_widget.set_image(image_array)

            time_end = time.time()
            
            print(f"update_image_display: {time_end - time_start:.2f} seconds")
                
        except Exception as e:
            
            print(f"画像表示の更新でエラー: {e}")
            traceback.print_exc()

    def _get_curve_array_from_points(self, points):
        """制御点のリストからカーブ配列を生成"""
        if len(points) < 2:
            return np.arange(65536, dtype=np.float32)
        
        sorted_points = sorted(points)
        x_points = [p[0] for p in sorted_points]
        y_points = [p[1] for p in sorted_points]
        
        x_curve = np.arange(65536, dtype=np.float32)
        interp = PchipInterpolator(x_points, y_points)
        y_curve = interp(x_curve)
        y_curve = np.clip(y_curve, 0, 65535)
        
        return y_curve

    def apply_parameters_to_editor(self, editor):
        """編集パラメータをエディターに適用"""
        if editor is None:
            return
        
        try:
            # エディタの状態を完全にリセット
            editor.reset()
            
            # 1. グローバル調整（マスクなし）を適用
            no_mask_key = self.tr("マスクなし")
            if no_mask_key in self.mask_edit_params:
                params = self.mask_edit_params[no_mask_key]
                
                editor.set_tone(
                    exposure=params.exposure,
                    contrast=params.contrast,
                    shadow=params.shadow,
                    highlight=params.highlight,
                    black=params.black,
                    white=params.white,
                    mask_name=None
                )
                editor.set_whitebalance_value(
                    temperature=params.wb_temperature,
                    tint=params.wb_tint,
                    mask_name=None
                )
                brightness_curve = self._get_curve_array_from_points(params.brightness_curve_points)
                editor.set_tone_curve(curve=brightness_curve, mask_name=None)
                
                oklch_h_curve = self._get_curve_array_from_points(params.oklch_h_curve_points)
                editor.set_oklch_h_tone_curve(curve=oklch_h_curve, mask_name=None)
                
                oklch_c_curve = self._get_curve_array_from_points(params.oklch_c_curve_points)
                editor.set_oklch_c_tone_curve(curve=(oklch_c_curve * 2), mask_name=None)

                oklch_l_curve = self._get_curve_array_from_points(params.oklch_l_curve_points)
                editor.set_oklch_l_tone_curve(curve=(oklch_l_curve * 2), mask_name=None)

                editor.set_vignette(strength=params.vignette, mask_name=None)

            # 2. 各マスクに対して個別に処理を適用
            for mask_name, params in self.mask_edit_params.items():
                if mask_name == no_mask_key:
                    continue
                
                # マスクがエディタに存在するかチェック
                if mask_name not in editor.mask_adjustments_values:
                    print(f"  警告: マスク '{mask_name}' がエディタに存在しません")
                    continue
                
                # マスクに対して調整を適用
                editor.set_tone(
                    exposure=params.exposure,
                    contrast=params.contrast,
                    shadow=params.shadow,
                    highlight=params.highlight,
                    black=params.black,
                    white=params.white,
                    mask_name=mask_name
                )
                
                editor.set_whitebalance_value(
                    temperature=params.wb_temperature,
                    tint=params.wb_tint,
                    mask_name=mask_name
                )
                
                # カーブを適用
                brightness_curve = self._get_curve_array_from_points(params.brightness_curve_points)
                editor.set_tone_curve(curve=brightness_curve, mask_name=mask_name)

                oklch_h_curve = self._get_curve_array_from_points(params.oklch_h_curve_points)
                editor.set_oklch_h_tone_curve(curve=oklch_h_curve, mask_name=mask_name)
                
                oklch_c_curve = self._get_curve_array_from_points(params.oklch_c_curve_points)
                editor.set_oklch_c_tone_curve(curve=(oklch_c_curve * 2), mask_name=mask_name)

                oklch_l_curve = self._get_curve_array_from_points(params.oklch_l_curve_points)
                editor.set_oklch_l_tone_curve(curve=(oklch_l_curve * 2), mask_name=mask_name)
                
                editor.set_vignette(strength=params.vignette, mask_name=mask_name)
                
                # マスク範囲を更新
                if mask_name in editor.mask_adjustments_values:
                    editor.mask_adjustments_values[mask_name]["mask_range_value"] = params.mask_range
                
        except Exception as e:
            print(f"パラメータ適用でエラー: {e}")
            traceback.print_exc()
    
    def update_metadata_display(self):
        """メタデータ表示を更新する (gridレイアウト・枠線対応版)"""
        if self.editor is None or self.editor.metadata is None:
            return

        try:
            # 既存のウィジェットをクリア
            for widget in self.metadata_frame.winfo_children():
                widget.destroy()

            # メタデータを取得
            metadata = self.editor.metadata.display_japanese("dict") if self.settings_manager.language == "日本語" else self.editor.metadata.metadata

            # ヘッダーを作成
            headings = [self.tr("キー"), self.tr("値")]
            for i, heading in enumerate(headings):
                label = ttk.Label(self.metadata_frame, text=heading, font=('TkDefaultFont', 10, 'bold'))
                label.grid(row=0, column=i, sticky='w', padx=5, pady=2)

            # ヘッダーの下に区切り線
            ttk.Separator(self.metadata_frame, orient='horizontal').grid(row=1, column=0, columnspan=2, sticky='ew')

            # データ行を作成
            grid_row = 2
            for key, value in metadata.items():
                # データ
                key_label = ttk.Label(self.metadata_frame, text=str(key), wraplength=400, justify="left")
                key_label.grid(row=grid_row, column=0, sticky='w', padx=5, pady=1)
                
                value_label = ttk.Label(self.metadata_frame, text=str(value), wraplength=500, justify="left")
                value_label.grid(row=grid_row, column=1, sticky='w', padx=5, pady=1)
                
                grid_row += 1
                
                # 各データ行の下に区切り線
                ttk.Separator(self.metadata_frame, orient='horizontal').grid(row=grid_row, column=0, columnspan=2, sticky='ew')
                grid_row += 1

            # 列の伸縮設定 (ウィンドウリサイズ時の挙動)
            self.metadata_frame.columnconfigure(0, weight=1)
            self.metadata_frame.columnconfigure(1, weight=2)

        except Exception as e:
            traceback.print_exc()
            print(f"メタデータ表示でエラー: {e}")
            
    # adjust_treeview_column_widths はgridレイアウトへの変更により不要になったため削除


    # 露出パラメータ変更ハンドラー
    def on_exposure_changed(self, value):
        self.edit_params.exposure = value
        self.update_image_display()
    
    def on_contrast_changed(self, value):
        self.edit_params.contrast = int(value)
        self.update_image_display()
    
    def on_shadow_changed(self, value):
        self.edit_params.shadow = int(value)
        self.update_image_display()
    
    def on_highlight_changed(self, value):
        self.edit_params.highlight = int(value)
        self.update_image_display()
    
    def on_black_changed(self, value):
        self.edit_params.black = int(value)
        self.update_image_display()
    
    def on_white_changed(self, value):
        self.edit_params.white = int(value)
        self.update_image_display()
    
    # ホワイトバランス変更ハンドラー
    def on_wb_temperature_changed(self, value):
        self.edit_params.wb_temperature = int(value)
        self.update_image_display()
    
    def on_wb_tint_changed(self, value):
        self.edit_params.wb_tint = int(value)
        self.update_image_display()
    
    def on_vignette_changed(self, value):
        self.edit_params.vignette = int(value)
        self.update_image_display()
    
    # トーンカーブ変更ハンドラー
    def on_brightness_curve_changed(self, points):
        """明るさカーブ変更"""
        self.edit_params.brightness_curve_points = points
        self.update_image_display()

    def on_oklch_h_curve_changed(self, control_points):
        """OKLCH Hカーブ変更"""
        self.edit_params.oklch_h_curve_points = control_points
        self.update_image_display()

    def on_oklch_c_curve_changed(self, control_points):
        """OKLCH Cカーブ変更"""
        self.edit_params.oklch_c_curve_points = control_points
        self.update_image_display()

    def on_oklch_l_curve_changed(self, control_points):
        """OKLCH Lカーブ変更"""
        self.edit_params.oklch_l_curve_points = control_points
        self.update_image_display()
    
    def on_mask_range_changed(self, value):
        """マスク範囲の微調整"""
        self.edit_params.mask_range = value
        self.update_image_display()
    
    def start_drag_timer(self):
        """ドラッグタイマーを開始"""
        self.dragging = True
        # 既存のタイマーをキャンセル
        if self.drag_timer_id:
            self.root.after_cancel(self.drag_timer_id)
        
        # 100ms後にドラッグ終了と判定
        self.drag_timer_id = self.root.after(100, self.on_drag_timeout)
        self.update_image_display()
    
    def on_drag_timeout(self):
        """ドラッグタイムアウト（高品質表示に切り替え）"""
        self.dragging = False
        self.drag_timer_id = None
        self.update_image_display(force_full=True)
    
    def reset_exposure_tab(self):
        """露出タブをリセット"""
        params = self.edit_params
        params.exposure = 0.0
        params.contrast = 0
        params.shadow = 0
        params.highlight = 0
        params.black = 0
        params.white = 0
        self.update_ui_from_parameters()
        self.update_image_display(force_full=True)
    
    def reset_wb_tab(self):
        """ホワイトバランスタブをリセット"""
        params = self.edit_params
        params.wb_temperature = 0
        params.wb_tint = 0
        self.update_ui_from_parameters()
        self.update_image_display(force_full=True)

    def reset_tone_curve_tab(self):
        """トーンカーブタブをリセット"""
        self.brightness_curve.reset_curve()
        self.oklch_h_curve.reset_curve()
        self.oklch_c_curve.reset_curve()
        self.oklch_l_curve.reset_curve()
        self.update_image_display(force_full=True)
    
    def reset_effects_tab(self):
        """効果タブをリセット"""
        params = self.edit_params
        params.vignette = 0
        self.update_ui_from_parameters()
        self.update_image_display(force_full=True)
    
    def reset_all_edits(self):
        """すべての編集をリセット"""
        # すべてのマスクの編集パラメータをリセット
        self.mask_edit_params = {name: EditParameters() for name in self.mask_edit_params}
        
        # UIを現在のマスクのパラメータで更新
        self.update_ui_from_parameters()
        self.update_image_display()
    
    def export_image(self):
        """画像エクスポート"""
        if self.editor is None:
            messagebox.showwarning(self.tr("警告"), self.tr("エクスポートする画像がありません"))
            return
        
        # エクスポート設定ダイアログ
        dialog = ExportDialog(self.root, self)
        if not dialog.show():
            return
        
        # ファイル名を生成
        if self.current_file_path:
            base_name = Path(self.current_file_path).stem
        else:
            base_name = "exported"
        
        format_str = dialog.get_format().lower()
        suggested_name = f"{base_name}_edited.{format_str}"
        
        # 保存先ダイアログ
        file_path = filedialog.asksaveasfilename(
            title=self.tr("画像をエクスポート"),
            initialfile=suggested_name,
            filetypes=[
                (f"{dialog.get_format()}", f"*.{format_str}"),
                (self.tr("すべてのファイル"), "*.*")
            ]
        )
        
        if not file_path:
            return
        
        # プログレスダイアログ
        progress = ProgressDialog(self.root, self.tr("エクスポート"), self.tr("画像をエクスポート中..."))

        # エクスポート処理
        def export_worker():
            time_start = time.time()
            try:
                # エディターの現在の状態を完全にリセット
                self.editor.reset()
                
                # 最終的なパラメータを適用
                self.apply_parameters_to_editor(self.editor)
                self.editor.apply_adjustments()

                time_end = time.time()
                print(f"Exported in {time_end - time_start:.2f} seconds")

                # エクスポート実行
                self.editor.save(file_path, quality=dialog.get_quality())
                
                # UIスレッドで結果を処理
                self.root.after(0, lambda: self.on_export_finished(file_path, progress))
                
            except Exception as e:
                traceback.print_exc()
                # UIスレッドでエラーを処理
                self.root.after(0, lambda: self.on_export_error(str(e), progress))
            
            
        
        # 別スレッドで実行
        threading.Thread(target=export_worker, daemon=True).start()
    
    def on_export_finished(self, export_path, progress):
        """エクスポート完了"""
        progress.close()
        messagebox.showinfo(
            self.tr("エクスポート完了"),
            f"{self.tr('画像を保存しました')}:{export_path}"
        )
    
    def on_export_error(self, error_message, progress):
        """エクスポートエラー"""
        progress.close()
        messagebox.showerror(self.tr("エラー"), f"{self.tr('エクスポートに失敗しました')}: {error_message}")
    
    def save_preset(self):
        """プリセット保存"""
        file_path = filedialog.asksaveasfilename(
            title=self.tr("プリセットを保存"),
            filetypes=[
                (self.tr("プリセットファイル"), "*.json"),
                (self.tr("すべてのファイル"), "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # 「マスクなし」の編集パラメータを辞書に変換
            preset_data = asdict(self.mask_edit_params[self.tr("マスクなし")])
            
            # JSONファイルに保存
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo(self.tr("情報"), self.tr("プリセットを保存しました"))
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror(self.tr("エラー"), f"{self.tr('プリセットの保存に失敗しました')}: {str(e)}")
    
    def load_preset(self):
        """プリセット読み込み"""
        file_path = filedialog.askopenfilename(
            title=self.tr("プリセットを読み込み"),
            filetypes=[
                (self.tr("プリセットファイル"), "*.json"),
                (self.tr("すべてのファイル"), "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                preset_data = {}
                for key, value in json.load(f).items():
                    if key == "hue_curve_points":
                        preset_data["oklch_h_curve_points"] = value
                        continue

                    elif key == "saturation_curve_points":
                        preset_data["oklch_c_curve_points"] = value
                        continue

                    elif key == "lightness_curve_points":
                        preset_data["oklch_l_curve_points"] = value
                        continue

                    preset_data[key] = value



            
            
             
            
            # 「マスクなし」のEditParametersオブジェクトを更新
            no_mask_key = self.tr("マスクなし")
            no_mask_params = self.mask_edit_params.setdefault(no_mask_key, EditParameters())
            for key, value in preset_data.items():
                if hasattr(no_mask_params, key):
                    setattr(no_mask_params, key, value)
            
            # 現在のUIが「マスクなし」の場合のみUIコントロールを更新
            if self.current_mask_name == no_mask_key:
                self.update_ui_from_parameters()
            
            # 画像表示は常に更新
            self.update_image_display()
            
            messagebox.showinfo(self.tr("情報"), self.tr("プリセットを読み込みました"))
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror(self.tr("エラー"), f"{self.tr('プリセットの読み込みに失敗しました')}: {str(e)}")
    
    def update_ui_from_parameters(self):
        """パラメータからUIを更新"""
        # 露出パラメータ
        self.exposure_slider.set_value(self.edit_params.exposure)
        self.contrast_slider.set_value(self.edit_params.contrast)
        self.shadow_slider.set_value(self.edit_params.shadow)
        self.highlight_slider.set_value(self.edit_params.highlight)
        self.black_slider.set_value(self.edit_params.black)
        self.white_slider.set_value(self.edit_params.white)

        # 周辺減光
        self.vignette_slider.set_value(self.edit_params.vignette)
        
        # ホワイトバランス
        self.wb_temperature_slider.set_value(self.edit_params.wb_temperature)
        self.wb_tint_slider.set_value(self.edit_params.wb_tint)
        
        # トーンカーブ
        self.brightness_curve.set_control_points(self.edit_params.brightness_curve_points)
        self.oklch_h_curve.set_control_points(self.edit_params.oklch_h_curve_points)
        self.oklch_c_curve.set_control_points(self.edit_params.oklch_c_curve_points)
        self.oklch_l_curve.set_control_points(self.edit_params.oklch_l_curve_points)

        # マスク範囲
        self.mask_range_slider.set_value(self.edit_params.mask_range)
    
    def show_settings(self):
        """設定ダイアログを表示"""
        settings_dialog = SettingsDialog(self.root, self.settings_manager, self)
        self.root.wait_window(settings_dialog.dialog)
    
    # マスク関連メソッド
    def start_ai_mask_creation(self):
        """AIマスク作成開始"""
        if self.editor is None:
            messagebox.showwarning(self.tr("警告"), self.tr("画像が読み込まれていません"))
            return
        
        self.image_widget.set_ai_mask_mode(True)
        self.ai_mask_btn.config(state=tk.DISABLED)
        self.cancel_mask_btn.config(state=tk.NORMAL)
    
    def cancel_ai_mask_creation(self):
        """AIマスク作成キャンセル"""
        self.image_widget.set_ai_mask_mode(False)
        self.ai_mask_btn.config(state=tk.NORMAL)
        self.cancel_mask_btn.config(state=tk.DISABLED)
    
    def on_image_clicked(self, x, y):
        """画像クリック処理"""
        if not self.image_widget.ai_mask_mode:
            return
        
        # AIマスク作成モードを終了
        self.cancel_ai_mask_creation()
        
        # マスク名を生成
        mask_count = len([name for name in self.editor.mask_adjustments_values.keys() if name.startswith(self.tr("マスク"))])
        mask_name = f"{self.tr('マスク')}{mask_count + 1}"
        
        # プログレスダイアログ
        progress = ProgressDialog(self.root, self.tr("AIマスク作成"), self.tr("AIマスク作成中..."))
        
        # AIマスクを生成する前に、すべての編集をフル解像度のエディターに適用する
        def ai_mask_worker():
            try:
                self.editor.reset()
                self.apply_parameters_to_editor(self.editor)
                self.editor.apply_adjustments()
                
                # AIマスク作成
                self.editor.create_ai_mask([x, y], mask_name=mask_name)
                
                # UIスレッドで結果を処理
                self.root.after(0, lambda: self.on_ai_mask_created(mask_name, progress))
                
            except Exception as e:
                traceback.print_exc()
                # UIスレッドでエラーを処理
                self.root.after(0, lambda: self.on_ai_mask_error(str(e), progress))
        
        # 別スレッドで実行
        threading.Thread(target=ai_mask_worker, daemon=True).start()
    
    def on_ai_mask_created(self, mask_name, progress):
        """AIマスク作成完了"""
        progress.close()
        
        # 新しいマスク用のパラメータを作成
        self.mask_edit_params[mask_name] = EditParameters()

        # マスクをリサイズ版のエディターにもコピー
        if mask_name in self.editor.mask_adjustments_values:
            mask_data = self.editor.mask_adjustments_values[mask_name]
            
            # リサイズ用にマスクをコピー（深いコピーを確実に実行）
            original_mask = mask_data["ndarray"].copy()  # 深いコピーを明示的に実行
            
            # 中解像度版
            if self.medium_editor:
                h_med, w_med = self.medium_editor.height, self.medium_editor.width
                print(original_mask.shape, h_med, w_med, original_mask.dtype)
                resized_mask_med = cv2.resize(original_mask, (w_med, h_med), interpolation=cv2.INTER_LINEAR)
                
                # 深いコピーでマスクデータを作成
                medium_mask_data = copy.deepcopy(mask_data)
                medium_mask_data["ndarray"] = resized_mask_med
                self.medium_editor.mask_adjustments_values[mask_name] = medium_mask_data
            
            # 低解像度版
            if self.small_editor:
                h_small, w_small = self.small_editor.height, self.small_editor.width
                resized_mask_small = cv2.resize(original_mask, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
                
                # 深いコピーでマスクデータを作成
                small_mask_data = copy.deepcopy(mask_data)
                small_mask_data["ndarray"] = resized_mask_small
                self.small_editor.mask_adjustments_values[mask_name] = small_mask_data
        
        # コンボボックスにマスクを追加
        current_values = list(self.mask_combo['values'])
        current_values.append(mask_name)
        self.mask_combo['values'] = current_values
        
        # 新しく作成したマスクを選択状態にし、UIの更新をトリガーする
        self.mask_var.set(mask_name)
        self.on_mask_selection_changed()
        
        messagebox.showinfo(
            self.tr("AIマスク作成完了"),
            f"{self.tr('AIマスクを作成しました')}:{mask_name}"
        )
    
    def on_ai_mask_error(self, error_message, progress):
        """AIマスク作成エラー"""
        progress.close()
        messagebox.showerror(self.tr("エラー"), f"{self.tr('AIマスクの作成に失敗しました')}: {error_message}")
    
    def _update_mask_controls_state(self):
        """現在のマスク選択状態に応じて、関連ウィジェットの有効/無効を切り替える"""
        is_mask_selected = self.current_mask_name != self.tr("マスクなし")

        # マスク関連ウィジェットの状態を更新
        self.delete_mask_btn.config(state=tk.NORMAL if is_mask_selected else tk.DISABLED)
        self.mask_display_check.config(state=tk.NORMAL if is_mask_selected else tk.DISABLED)
        self.invert_mask_btn.config(state=tk.NORMAL if is_mask_selected else tk.DISABLED)
        self.mask_range_slider.slider.config(state=tk.NORMAL if is_mask_selected else tk.DISABLED)
        self.mask_range_slider.spinbox.config(state=tk.NORMAL if is_mask_selected else tk.DISABLED)

        # 周辺減光スライダーの状態を更新
        if hasattr(self, 'vignette_slider'):
            if is_mask_selected:
                self.vignette_slider.slider.config(state='disabled')
                self.vignette_slider.spinbox.config(state='disabled')
                
            else:
                self.vignette_slider.slider.config(state='normal')
                self.vignette_slider.spinbox.config(state='normal')

        if is_mask_selected:
            self.mask_display_var.set(False)
            # マスク表示がオフになったことを反映
            if self.dragging:
                self.update_image_display()
            else:
                self.update_image_display(force_full=True)

    def on_mask_selection_changed(self, event=None):
        """マスク選択変更"""
        self.current_mask_name = self.mask_var.get()
        self.update_ui_from_parameters()
        self._update_mask_controls_state()
    
    def delete_current_mask(self):
        """現在のマスクを削除"""
        if self.current_mask_name == self.tr("マスクなし"):
            return
        
        # 確認ダイアログ
        result = messagebox.askyesno(
            self.tr("確認"),
            f"{self.tr('マスク')} '{self.current_mask_name}' {self.tr('を削除しますか？')}"
        )
        
        if not result:
            return
        
        mask_to_delete = self.current_mask_name

        # パラメータとエディタ内のマスクを削除
        if mask_to_delete in self.mask_edit_params:
            del self.mask_edit_params[mask_to_delete]
        if self.editor and mask_to_delete in self.editor.mask_adjustments_values:
            del self.editor.mask_adjustments_values[mask_to_delete]
        if self.medium_editor and mask_to_delete in self.medium_editor.mask_adjustments_values:
            del self.medium_editor.mask_adjustments_values[mask_to_delete]
        if self.small_editor and mask_to_delete in self.small_editor.mask_adjustments_values:
            del self.small_editor.mask_adjustments_values[mask_to_delete]
        
        # コンボボックスから削除
        current_values = list(self.mask_combo['values'])
        if mask_to_delete in current_values:
            current_values.remove(mask_to_delete)
            self.mask_combo['values'] = current_values
        
        # "マスクなし" を選択状態にし、UIの更新をトリガーする
        self.mask_var.set(self.tr("マスクなし"))
        self.on_mask_selection_changed()
    
    def invert_current_mask(self):
        """現在のマスクを反転して新規作成"""
        if self.current_mask_name is None or self.current_mask_name == self.tr("マスクなし"):
            messagebox.showwarning(self.tr("警告"), self.tr("反転するマスクが選択されていません"))
            return
        
        if self.editor is None or self.current_mask_name not in self.editor.mask_adjustments_values:
            return
        
        # 新しいマスク名を生成
        mask_count = len([name for name in self.editor.mask_adjustments_values.keys() if name.startswith(self.tr("マスク"))])
        new_mask_name = f"{self.tr('マスク')}{mask_count + 1}"
        
        # 元のマスクデータを深いコピーで取得
        original_mask_data = copy.deepcopy(self.editor.mask_adjustments_values[self.current_mask_name])
        
        # マスクを反転
        original_mask_data["ndarray"] = original_mask_data["ndarray"] * -1.0
        original_mask_data["inverted"] = not original_mask_data.get("inverted", False)
        
        # 新しいマスクを作成
        self.editor.mask_adjustments_values[new_mask_name] = original_mask_data
        
        # リサイズ版にもコピー（各々で独立したデータを作成）
        if self.medium_editor:
            medium_mask_data = copy.deepcopy(original_mask_data)
            h_med, w_med = self.medium_editor.height, self.medium_editor.width
            resized_mask = cv2.resize(original_mask_data["ndarray"], (w_med, h_med), interpolation=cv2.INTER_LINEAR)
            medium_mask_data["ndarray"] = resized_mask
            self.medium_editor.mask_adjustments_values[new_mask_name] = medium_mask_data
        
        if self.small_editor:
            small_mask_data = copy.deepcopy(original_mask_data)
            h_small, w_small = self.small_editor.height, self.small_editor.width
            resized_mask = cv2.resize(original_mask_data["ndarray"], (w_small, h_small), interpolation=cv2.INTER_LINEAR)
            small_mask_data["ndarray"] = resized_mask
            self.small_editor.mask_adjustments_values[new_mask_name] = small_mask_data
        
        # 新しいマスク用にデフォルトのパラメータを作成
        new_params = EditParameters()
        # 元のマスクのマスク範囲設定のみを反転して引き継ぐ
        new_params.mask_range = -self.edit_params.mask_range
        self.mask_edit_params[new_mask_name] = new_params

        # コンボボックスに追加し、それを選択状態にする
        current_values = list(self.mask_combo['values'])
        current_values.append(new_mask_name)
        self.mask_combo['values'] = current_values
        self.mask_var.set(new_mask_name)
        self.current_mask_name = new_mask_name
        self.update_ui_from_parameters()
        
        messagebox.showinfo(
            self.tr("マスク作成完了"),
            f"{self.tr('反転マスクを作成しました')}:{new_mask_name}"
        )
    
    def on_mask_display_toggled(self):
        """マスク表示切り替え"""
        self.mask_display_enabled = self.mask_display_var.get()
        self.update_image_display()
    
    def run(self):
        """アプリケーションを実行"""
        # コマンドライン引数でファイルが渡された場合
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            if os.path.exists(file_path):
                # UIのイベントループが開始してから実行されるよう少し遅延
                self.root.after(100, lambda: self.load_image(file_path))
        
        self.root.mainloop()


def main():
    """メイン関数"""
    # メインウィンドウ作成
    app = RAWDevelopmentGUI()
    app.run()

if __name__ == "__main__":
    main()
