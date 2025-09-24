"RAW現像ソフトメインUI"

import sys
import os
import json
import copy
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import traceback

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QPushButton, QTabWidget,
    QMenuBar, QMenu, QFileDialog, QMessageBox, QProgressDialog,
    QComboBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QScrollArea, QFrame, QDialog, QDialogButtonBox, QGridLayout,
    QSplitter, QHeaderView, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, QTimer, Signal, QTranslator, QLocale, QRect, QPoint
from PySide6.QtGui import QPixmap, QImage, QPainter, QCursor, QAction, QDragEnterEvent, QDropEvent
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from scipy.interpolate import PchipInterpolator
import colorsys
try:
    import raw_image_editor
    import photo_metadata
    RAW_EDITOR_AVAILABLE = True
except ImportError:
    RAW_EDITOR_AVAILABLE = False
    print("Warning: raw_image_editor or photo_metadata not available")

# 定数
DRAG_IMAGE_SIZE = 100  # ドラッグ中の画像サイズ（長辺）
NORMAL_IMAGE_SIZE = 800  # 通常時の画像サイズ（長辺）

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
    hue_curve_points: List[Tuple[float, float]] = None
    saturation_curve_points: List[Tuple[float, float]] = None
    lightness_curve_points: List[Tuple[float, float]] = None

    # マスク範囲
    mask_range: float = 0.0
    
    def __post_init__(self):
        if self.brightness_curve_points is None:
            self.brightness_curve_points = [(0, 0), (65535, 65535)]
        if self.hue_curve_points is None:
            self.hue_curve_points = [(0, 0), (65535, 65535)]
        if self.saturation_curve_points is None:
            self.saturation_curve_points = [(0, 32767.5), (65535, 32767.5)]
        if self.lightness_curve_points is None:
            self.lightness_curve_points = [(0, 32767.5), (65535, 32767.5)]

@dataclass
class SettingsManager:
    """設定とパスを管理するデータクラス"""
    device: str = "cpu"
    language: str = "日本語"
    
    # パス関連（初期化後に設定）
    app_dir: Path = None
    settings_path: Path = None
    exiftool_path: str = None
    language_path: Path = None

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
        self.language_path = self.app_dir / "languages.json"

    def load(self):
        """設定をJSONファイルから読み込む"""
        if not self.settings_path.exists():
            return
        try:
            with open(self.settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            
            # デバイス設定の読み込みと検証
            device_setting = settings.get("device", self.device)
            if device_setting.startswith("opencl:"):
                if RAW_EDITOR_AVAILABLE and raw_image_editor.OPENCL_AVAILABLE:
                    try:
                        opencl_contexts = raw_image_editor.get_opencl_ctxs()
                        device_index = int(device_setting.split(":")[1])
                        if device_index < len(opencl_contexts):
                            self.device = device_setting
                        else:
                            self.device = "cpu"  # インデックスが範囲外
                    except (ValueError, IndexError, Exception):
                        self.device = "cpu"  # パースエラーやOpenCLエラー
                else:
                    self.device = "cpu"  # OpenCLが利用不可
            else:
                self.device = device_setting  # "cpu"

            self.language = settings.get("language", self.language)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"設定ファイルの読み込みに失敗しました: {e}")

    def save(self):
        """現在の設定をJSONファイルに保存する"""
        settings = {
            "device": self.device,
            "language": self.language
        }
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"設定の保存に失敗しました: {e}")
            raise e

class ToneCurveWidget(FigureCanvas):
    """matplotlibを使ったトーンカーブエディタ"""
    
    curve_changed = Signal(list)  # カーブが変更された時のシグナル
    
    def __init__(self, curve_type="brightness"):
        self.fig = Figure(figsize=(4, 4), facecolor='none')
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        super().__init__(self.fig)
        
        self.curve_type = curve_type
        self.ax = self.fig.add_subplot(111)
        self.histogram_data = None  # ヒストグラムデータを保持
        self.setup_plot()
        
        # 初期制御点
        if curve_type == "brightness" or curve_type == "hue":
            self.control_points = [(0, 0), (65535, 65535)]
        else:  # saturation, lightness
            self.control_points = [(0, 32767.5), (65535, 32767.5)]
            
        self.selected_point = None
        self.dragging = False
        
        # イベント接続
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)
        
        self.update_plot()
    
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
                    self.ax.plot(x_values, self.histogram_data['white'] * scale, color='white', alpha=0.6, linewidth=1)
                if 'r' in self.histogram_data and self.histogram_data['r'] is not None:
                    self.ax.plot(x_values, self.histogram_data['r'] * scale, color='red', alpha=0.6, linewidth=1)
                if 'g' in self.histogram_data and self.histogram_data['g'] is not None:
                    self.ax.plot(x_values, self.histogram_data['g'] * scale, color='green', alpha=0.6, linewidth=1)
                if 'b' in self.histogram_data and self.histogram_data['b'] is not None:
                    self.ax.plot(x_values, self.histogram_data['b'] * scale, color='blue', alpha=0.6, linewidth=1)
            
        elif self.curve_type == "hue":
            # 色相のグラデーション背景とカラーライン
            hue_gradient = np.linspace(0, 1, 256).reshape(1, 256)
            hue_gradient = np.repeat(hue_gradient, 256, axis=0)
            colors = plt.cm.hsv(hue_gradient)
            self.ax.imshow(colors, extent=[0, 65535, 0, 65535], aspect='auto', alpha=0.6)
            
            # カラーライン
            y_pos = np.linspace(0, 65535, 7)
            colors_line = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'red']
            for y, color in zip(y_pos, colors_line):
                self.ax.axhline(y=y, color=color, linewidth=2, alpha=0.8)
                
                
                
        elif self.curve_type == "saturation":
            # 彩度のグラデーション背景
            x = np.linspace(0, 1, 256)
            y = np.linspace(0, 1, 256)
            X, Y = np.meshgrid(x, y)
            
            # HSLカラースペースで彩度グラデーション
            H = X  # 色相は横方向
            S = Y  # 彩度は縦方向
            L = np.full_like(X, 0.5)  # 輝度は固定
            
            colors = np.zeros((256, 256, 3))
            for i in range(256):
                for j in range(256):
                    colors[i, j] = colorsys.hls_to_rgb(H[i, j], L[i, j], S[i, j])
            
            self.ax.imshow(colors, extent=[0, 65535, 0, 65535], aspect='auto', alpha=0.7, origin='lower')
            
        elif self.curve_type == "lightness":
            # 輝度のグラデーション背景
            x = np.linspace(0, 1, 256)
            y = np.linspace(0, 1, 256)
            X, Y = np.meshgrid(x, y)
            
            # HSLカラースペースで輝度グラデーション
            H = X  # 色相は横方向
            L = Y  # 輝度は縦方向
            S = np.full_like(X, 0.8)  # 彩度は固定
            
            colors = np.zeros((256, 256, 3))
            for i in range(256):
                for j in range(256):
                    colors[i, j] = colorsys.hls_to_rgb(H[i, j], L[i, j], S[i, j])
            
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
                    if self.curve_type == "brightness" or self.curve_type == "hue":
                        self.ax.plot([0, 65535], [0, 65535], 'b--', alpha=0.5, linewidth=1)
                    else:  # saturation, lightness
                        self.ax.plot([0, 65535], [32767.5, 32767.5], 'b--', alpha=0.5, linewidth=1)
        
        # 制御点をプロット
        for i, (x, y) in enumerate(self.control_points):
            self.ax.plot(x, y, 'ro', markeredgecolor='black', markeredgewidth=2, markersize=8)
        
        self.draw()
    
    def is_straight_line(self):
        """直線かどうかを判定"""
        if len(self.control_points) <= 2:
            if self.curve_type == "brightness" or self.curve_type == "hue":
                return self.control_points == [(0, 0), (65535, 65535)]
            else:  # saturation, lightness
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
        self.dragging = False
        self.selected_point = None
    
    def emit_curve_changed(self):
        """カーブ変更シグナルを発信"""
        self.curve_changed.emit(self.control_points.copy())

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
            if self.curve_type == "brightness" or self.curve_type == "hue":
                self.control_points = [(0, 0), (65535, 65535)]
            else:  # saturation, lightness
                self.control_points = [(0, 32767.5), (65535, 32767.5)]
        self.update_plot()
    
    def reset_curve(self):
        """カーブをリセット"""
        if self.curve_type == "brightness" or self.curve_type == "hue":
            self.control_points = [(0, 0), (65535, 65535)]
        else:  # saturation, lightness
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

class SliderWithSpinBox(QWidget):
    """スライダーと数値入力ボックスを組み合わせたウィジェット"""
    
    valueChanged = Signal(float)
    
    def __init__(self, min_val, max_val, step, decimals=0, initial_value=0):
        super().__init__()
        
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.decimals = decimals
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # スライダー
        self.slider = QSlider(Qt.Horizontal)
        slider_min = int(min_val / step)
        slider_max = int(max_val / step)
        self.slider.setMinimum(slider_min)
        self.slider.setMaximum(slider_max)
        self.slider.setValue(int(initial_value / step))
        
        # 数値入力ボックス
        if decimals > 0:
            self.spinbox = QDoubleSpinBox()
            self.spinbox.setDecimals(decimals)
        else:
            self.spinbox = QSpinBox()
        
        self.spinbox.setMinimum(min_val)
        self.spinbox.setMaximum(max_val)
        self.spinbox.setSingleStep(step)
        self.spinbox.setValue(initial_value)
        
        layout.addWidget(self.slider, 3)
        layout.addWidget(self.spinbox, 1)
        self.setLayout(layout)
        
        # シグナル接続
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spinbox.valueChanged.connect(self.on_spinbox_changed)
    
    def on_slider_changed(self, value):
        """スライダー値変更"""
        real_value = value * self.step
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(real_value)
        self.spinbox.blockSignals(False)
        self.valueChanged.emit(real_value)
    
    def on_spinbox_changed(self, value):
        """スピンボックス値変更"""
        slider_value = int(value / self.step)
        self.slider.blockSignals(True)
        self.slider.setValue(slider_value)
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)
    
    def setValue(self, value):
        """値を設定"""
        self.slider.blockSignals(True)
        self.spinbox.blockSignals(True)
        
        slider_value = int(value / self.step)
        self.slider.setValue(slider_value)
        self.spinbox.setValue(value)
        
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)
    
    def value(self):
        """現在の値を取得"""
        return self.spinbox.value()

class ImageDisplayWidget(QLabel):
    """画像表示ウィジェット"""
    
    clicked = Signal(int, int)  # クリック位置を通知
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 1px solid gray;")
        self.setAlignment(Qt.AlignCenter)
        self.setText(self.main_window.tr("画像をドラッグ&ドロップまたは「ファイル」→「写真を開く」"))
        self.setAcceptDrops(True)
        self.original_image = None
        self.ai_mask_mode = False
        self.full_res_width = 1
        self.full_res_height = 1
    
    def mousePressEvent(self, event):
        """マウスクリック処理"""
        if event.button() == Qt.LeftButton and self.main_window.medium_editor:
            # まず初期画像を表示
            self.set_image(self.main_window.medium_editor.initial_image_array)

            # AIマスクモードの場合、クリック位置を通知
            if self.ai_mask_mode:
                if self.pixmap() and not self.pixmap().isNull():
                    pixmap = self.pixmap()
                    
                    # 画像の表示領域を計算 (アスペクト比維持による余白を考慮)
                    img_rect = QRect(QPoint(0, 0), pixmap.size())
                    img_rect.moveCenter(self.rect().center())

                    # クリック位置が画像内かチェック
                    if img_rect.contains(event.position().toPoint()):
                        # 画像内での相対座標を計算
                        relative_pos = event.position().toPoint() - img_rect.topLeft()
                        
                        # 0-1の範囲に正規化
                        norm_x = relative_pos.x() / img_rect.width()
                        norm_y = relative_pos.y() / img_rect.height()
                        
                        # フル解像度の座標に変換
                        full_res_x = int(norm_x * self.full_res_width)
                        full_res_y = int(norm_y * self.full_res_height)
                        
                        self.clicked.emit(full_res_x, full_res_y)

    def mouseReleaseEvent(self, event):
        """マウスリリース処理"""
        if event.button() == Qt.LeftButton and self.main_window.editor:
            # 通常の表示に更新
            self.main_window.update_image_display(force_full=True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """ドラッグエンター処理"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """ドロップ処理"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path:
                self.main_window.load_image(file_path)
    
    def set_image(self, image_array, is_mask_overlay=False):
        """画像を設定"""
        if image_array is None:
            return
        
        # numpy配列をQImageに変換
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        h, w = image_array.shape[:2]
        
        if len(image_array.shape) == 3:  # カラー画像
            qimage = QImage(image_array.data, w, h, w * 3, QImage.Format_RGB888)
        else:  # グレースケール
            qimage = QImage(image_array.data, w, h, w, QImage.Format_Grayscale8)
        
        # ウィジェットサイズに合わせてスケール
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)
        
        if not is_mask_overlay:
            self.original_image = qimage
    
    def set_ai_mask_mode(self, enabled):
        """AIマスクモードの設定"""
        self.ai_mask_mode = enabled
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

class SquareWrapper(QWidget):
    """子ウィジェットを常に正方形に保つラッパーウィジェット"""
    def __init__(self, child_widget):
        super().__init__()
        self.child = child_widget
        self.child.setParent(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = event.size().width()
        h = event.size().height()
        size = min(w, h)
        
        # 子ウィジェットを中央に配置
        x = (w - size) // 2
        y = (h - size) // 2
        
        self.child.setGeometry(x, y, size, size)

class ImageLoadWorker(QThread):
    """画像読み込み処理用のワーカースレッド"""
    
    finished = Signal(object, object, object)  # editor, medium_editor, small_editor
    error = Signal(str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        """画像読み込み処理"""
        try:
            if not RAW_EDITOR_AVAILABLE:
                self.error.emit("raw_image_editor が利用できません")
                return
            
            # 画像を読み込み
            editor = raw_image_editor.RAWImageEditor(file_path=self.file_path, apply_clahe=True, lens_correction=True, gamma=(2.222, 4.5))
            
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
            medium_editor = self.create_resized_editor(editor, NORMAL_IMAGE_SIZE)
            small_editor = self.create_resized_editor(editor, DRAG_IMAGE_SIZE)
            
            self.finished.emit(editor, medium_editor, small_editor)
            
        except Exception as e:
            self.error.emit(f"画像の読み込みに失敗しました: {str(e)}")
    
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

class ExportWorker(QThread):
    """画像エクスポート処理用のワーカースレッド"""
    
    finished = Signal(str)  # エクスポート先パス
    error = Signal(str)
    
    def __init__(self, editor, export_path, quality):
        super().__init__()
        self.editor = editor
        self.export_path = export_path
        self.quality = quality
    
    def run(self):
        """エクスポート処理"""
        try:
            self.editor.save(self.export_path, quality=self.quality)
            self.finished.emit(self.export_path)
        except Exception as e:
            self.error.emit(f"エクスポートに失敗しました: {str(e)}")

class AIMaskWorker(QThread):
    """AIマスク作成用のワーカースレッド"""
    
    finished = Signal(str)  # マスク名
    error = Signal(str)
    
    def __init__(self, editor, x, y, mask_name):
        super().__init__()
        self.editor = editor
        self.x = x
        self.y = y
        self.mask_name = mask_name
    
    def run(self):
        """AIマスク作成処理"""
        try:
            self.editor.create_ai_mask([self.x, self.y], mask_name=self.mask_name)
            self.finished.emit(self.mask_name)
        except Exception as e:
            self.error.emit(f"AIマスクの作成に失敗しました: {str(e)}")

class ExportDialog(QDialog):
    """エクスポート設定ダイアログ"""
    
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle(self.main_window.tr("エクスポート設定"))
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # 画質設定
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel(self.main_window.tr("画質:")))
        self.quality_combo = QComboBox()
        for q in range(30, 101, 5):
            self.quality_combo.addItem(str(q))
        self.quality_combo.setCurrentText("90")
        quality_layout.addWidget(self.quality_combo)
        layout.addLayout(quality_layout)
        
        # 形式設定
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel(self.main_window.tr("形式:")))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["JPEG", "PNG"])
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)
        
        # ボタン
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton(self.main_window.tr("キャンセル"))
        export_btn = QPushButton(self.main_window.tr("エクスポート"))
        
        cancel_btn.clicked.connect(self.reject)
        export_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(export_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_quality(self):
        return int(self.quality_combo.currentText())
    
    def get_format(self):
        return self.format_combo.currentText()

class SettingsDialog(QDialog):
    """設定ダイアログ"""
    
    def __init__(self, settings_manager: SettingsManager, main_window, parent=None):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.main_window = main_window
        self.setWindowTitle(self.main_window.tr("設定"))
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # タブウィジェット
        tab_widget = QTabWidget()
        
        # 処理デバイスタブ
        device_tab = QWidget()
        device_layout = QVBoxLayout()
        
        device_layout.addWidget(QLabel(self.main_window.tr("処理デバイス:")))
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU (numpy)")
        
        # OpenCLデバイス一覧を取得
        if RAW_EDITOR_AVAILABLE and raw_image_editor.OPENCL_AVAILABLE:
            try:
                opencl_contexts = raw_image_editor.get_opencl_ctxs()
                for i, ctx in enumerate(opencl_contexts):
                    device_name = ctx.devices[0].name
                    self.device_combo.addItem(f"OpenCL: {device_name}")
            except Exception:
                pass
        
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        device_tab.setLayout(device_layout)
        
        # 言語タブ
        language_tab = QWidget()
        language_layout = QVBoxLayout()
        
        language_layout.addWidget(QLabel(self.main_window.tr("言語:")))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "日本語"])
        
        language_layout.addWidget(self.language_combo)
        language_layout.addStretch()
        language_tab.setLayout(language_layout)
        
        tab_widget.addTab(device_tab, self.main_window.tr("処理デバイス"))
        tab_widget.addTab(language_tab, self.main_window.tr("言語"))
        
        layout.addWidget(tab_widget)
        
        # 保存ボタン
        save_btn = QPushButton(self.main_window.tr("設定を保存"))
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        
        self.setLayout(layout)
        
        # 設定読み込み
        self.load_settings()
    
    def load_settings(self):
        """設定を読み込みUIに反映"""
        device = self.settings_manager.device
        if device.startswith("opencl:"):
            try:
                index = int(device.split(":")[1])
                if 0 <= index < self.device_combo.count() - 1:
                    self.device_combo.setCurrentIndex(index + 1)
            except (ValueError, IndexError):
                pass  # 不正な値ならデフォルト
        
        language = self.settings_manager.language
        self.language_combo.setCurrentText(language)
    
    def save_settings(self):
        """UIから設定を取得し保存"""
        device_index = self.device_combo.currentIndex()
        if device_index == 0:
            device = "cpu"
        else:
            device = f"opencl:{device_index - 1}"
        
        self.settings_manager.device = device
        self.settings_manager.language = self.language_combo.currentText()
        
        try:
            self.settings_manager.save()
            QMessageBox.information(self, self.main_window.tr("情報"), self.main_window.tr("設定を保存しました。ソフトウェアを再起動してください。"))
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, self.main_window.tr("警告"), f"{self.main_window.tr('設定の保存に失敗しました')}: {str(e)}")

class RAWDevelopmentGUI(QMainWindow):
    """RAW現像ソフトのメインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        
        # 設定マネージャーを初期化し、設定を読み込む
        self.settings_manager = SettingsManager()
        self.settings_manager.__post_init__()
        self.settings_manager.load()

        # 翻訳を読み込む
        self.translations = {}
        self.load_translations()

        # 初期化
        self.editor = None
        self.medium_editor = None
        self.small_editor = None
        self.current_file_path = None
        self.device = self.settings_manager.device
        self.dragging = False
        self.mask_display_enabled = False

        # raw_image_editorの初期化
        if RAW_EDITOR_AVAILABLE:
            try:
                raw_image_editor.photo_metadata.set_exiftool_path(self.settings_manager.exiftool_path)
                raw_image_editor.init(device=self.device)
            except Exception as e:
                print(f"raw_image_editor の初期化に失敗: {e}")
                QMessageBox.warning(self, self.tr("初期化エラー"), f"{self.tr('画像処理エンジンの初期化に失敗しました。')}\n{e}")
        
        # 編集パラメータ（マスクごと）
        self.mask_edit_params: Dict[str, EditParameters] = {self.tr("マスクなし"): EditParameters()}
        self.current_mask_name = self.tr("マスクなし")
        
        # UI初期化
        self.init_ui()
        
        # ドラッグ検出用のタイマー
        self.drag_timer = QTimer()
        self.drag_timer.timeout.connect(self.on_drag_timeout)
        self.drag_timer.setSingleShot(True)
    
    def tr(self, key: str) -> str:
        """指定されたキーの翻訳済みテキストを返す"""
        return self.translations.get(self.settings_manager.language, {}).get(key, key)

    def load_translations(self):
        """言語ファイルを読み込む"""
        if not self.settings_manager.language_path.exists():
            return
        try:
            with open(self.settings_manager.language_path, "r", encoding="utf-8") as f:
                self.translations = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"言語ファイルの読み込みに失敗しました: {e}")
            self.translations = {}

    @property
    def edit_params(self) -> EditParameters:
        """現在のマスクに対応する編集パラメータを取得"""
        return self.mask_edit_params.setdefault(self.current_mask_name, EditParameters())

    def init_ui(self):
        """UI初期化"""
        self.setWindowTitle(self.tr("RAW現像ソフト"))
        self.setGeometry(100, 100, 1400, 900)
        
        # メニューバー
        self.create_menu_bar()
        
        # メインウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # メインレイアウト（横分割）
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # スプリッター
        splitter = QSplitter(Qt.Horizontal)
        
        # 左側：画像表示（60%）
        self.image_widget = ImageDisplayWidget(self)
        self.image_widget.clicked.connect(self.on_image_clicked)
        splitter.addWidget(self.image_widget)
        
        # 右側：編集パネル（40%）
        self.edit_panel = self.create_edit_panel()
        splitter.addWidget(self.edit_panel)
        
        # 分割比率設定
        splitter.setSizes([840, 560])  # 60:40の比率
        
        main_layout.addWidget(splitter)
    
    def create_menu_bar(self):
        """メニューバー作成"""
        menubar = self.menuBar()
        
        # ファイルメニュー
        file_menu = menubar.addMenu(self.tr("ファイル"))
        
        open_action = QAction(self.tr("写真を開く"), self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        export_action = QAction(self.tr("写真をエクスポート"), self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_image)
        file_menu.addAction(export_action)
        
        # 編集メニュー
        edit_menu = menubar.addMenu(self.tr("編集"))
        
        reset_action = QAction(self.tr("すべての編集をリセット"), self)
        reset_action.triggered.connect(self.reset_all_edits)
        edit_menu.addAction(reset_action)
        
        # プリセットメニュー
        preset_menu = menubar.addMenu(self.tr("プリセット"))
        
        save_preset_action = QAction(self.tr("現在の編集をプリセットとして保存"), self)
        save_preset_action.triggered.connect(self.save_preset)
        preset_menu.addAction(save_preset_action)
        
        load_preset_action = QAction(self.tr("プリセットを読み込み"), self)
        load_preset_action.triggered.connect(self.load_preset)
        preset_menu.addAction(load_preset_action)
        
        # 設定メニュー
        settings_menu = menubar.addMenu(self.tr("設定"))
        
        settings_action = QAction(self.tr("設定"), self)
        settings_action.triggered.connect(self.show_settings)
        settings_menu.addAction(settings_action)
    
    def create_edit_panel(self):
        """編集パネル作成"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # タブウィジェット
        self.tab_widget = QTabWidget()
        
        # 各タブを作成
        self.create_exposure_tab()
        self.create_tone_curve_tabs()
        self.create_wb_tab()
        self.create_mask_tab()
        self.create_metadata_tab()
        
        layout.addWidget(self.tab_widget)
        
        return panel
    
    def create_exposure_tab(self):
        """露出タブ作成"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # 説明
        layout.addWidget(QLabel(self.tr("基本的な露出調整を行います")))
        
        # 露出
        layout.addWidget(QLabel(self.tr("露出:")))
        self.exposure_slider = SliderWithSpinBox(-6.0, 6.0, 0.05, 2, 0.0)
        self.exposure_slider.valueChanged.connect(self.on_exposure_changed)
        layout.addWidget(self.exposure_slider)
        
        # コントラスト
        layout.addWidget(QLabel(self.tr("コントラスト:")))
        self.contrast_slider = SliderWithSpinBox(-100, 100, 1, 0, 0)
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        layout.addWidget(self.contrast_slider)
        
        # シャドウ
        layout.addWidget(QLabel(self.tr("シャドウ:")))
        self.shadow_slider = SliderWithSpinBox(-100, 100, 1, 0, 0)
        self.shadow_slider.valueChanged.connect(self.on_shadow_changed)
        layout.addWidget(self.shadow_slider)
        
        # ハイライト
        layout.addWidget(QLabel(self.tr("ハイライト:")))
        self.highlight_slider = SliderWithSpinBox(-100, 100, 1, 0, 0)
        self.highlight_slider.valueChanged.connect(self.on_highlight_changed)
        layout.addWidget(self.highlight_slider)
        
        # 黒レベル
        layout.addWidget(QLabel(self.tr("黒レベル:")))
        self.black_slider = SliderWithSpinBox(-100, 100, 1, 0, 0)
        self.black_slider.valueChanged.connect(self.on_black_changed)
        layout.addWidget(self.black_slider)
        
        # 白レベル
        layout.addWidget(QLabel(self.tr("白レベル:")))
        self.white_slider = SliderWithSpinBox(-100, 100, 1, 0, 0)
        self.white_slider.valueChanged.connect(self.on_white_changed)
        layout.addWidget(self.white_slider)
        
        layout.addStretch()
        
        # リセットボタン
        reset_btn = QPushButton(self.tr("このタブをリセット"))
        reset_btn.clicked.connect(self.reset_exposure_tab)
        layout.addWidget(reset_btn)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, self.tr("露出"))
    
    def create_tone_curve_tabs(self):
        """トーンカーブタブ群作成"""
        # 明るさカーブ
        self.create_brightness_curve_tab()
        
        # 色相カーブ
        self.create_hue_curve_tab()
        
        # 彩度カーブ
        self.create_saturation_curve_tab()
        
        # 輝度カーブ
        self.create_lightness_curve_tab()
    
    def create_brightness_curve_tab(self):
        """明るさカーブタブ作成"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel(self.tr("明るさのトーンカーブを調整します")))
        
        # トーンカーブウィジェット
        self.brightness_curve = ToneCurveWidget("brightness")
        self.brightness_curve.curve_changed.connect(self.on_brightness_curve_changed)
        wrapper = SquareWrapper(self.brightness_curve)
        layout.addWidget(wrapper)
        
        # リセットボタン
        reset_btn = QPushButton(self.tr("トーンカーブをリセット"))
        reset_btn.clicked.connect(self.brightness_curve.reset_curve)
        layout.addWidget(reset_btn)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, self.tr("明るさカーブ"))
    
    def create_hue_curve_tab(self):
        """色相カーブタブ作成"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel(self.tr("色相のトーンカーブを調整します")))
        
        self.hue_curve = ToneCurveWidget("hue")
        self.hue_curve.curve_changed.connect(self.on_hue_curve_changed)
        wrapper = SquareWrapper(self.hue_curve)
        layout.addWidget(wrapper)
        
        reset_btn = QPushButton(self.tr("トーンカーブをリセット"))
        reset_btn.clicked.connect(self.hue_curve.reset_curve)
        layout.addWidget(reset_btn)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, self.tr("色相カーブ"))
    
    def create_saturation_curve_tab(self):
        """彩度カーブタブ作成"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel(self.tr("彩度のトーンカーブを調整します")))
        
        self.saturation_curve = ToneCurveWidget("saturation")
        self.saturation_curve.curve_changed.connect(self.on_saturation_curve_changed)
        wrapper = SquareWrapper(self.saturation_curve)
        layout.addWidget(wrapper)
        
        reset_btn = QPushButton(self.tr("トーンカーブをリセット"))
        reset_btn.clicked.connect(self.saturation_curve.reset_curve)
        layout.addWidget(reset_btn)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, self.tr("彩度カーブ"))
    
    def create_lightness_curve_tab(self):
        """輝度カーブタブ作成"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel(self.tr("輝度のトーンカーブを調整します")))
        
        self.lightness_curve = ToneCurveWidget("lightness")
        self.lightness_curve.curve_changed.connect(self.on_lightness_curve_changed)
        wrapper = SquareWrapper(self.lightness_curve)
        layout.addWidget(wrapper)
        
        reset_btn = QPushButton(self.tr("トーンカーブをリセット"))
        reset_btn.clicked.connect(self.lightness_curve.reset_curve)
        layout.addWidget(reset_btn)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, self.tr("輝度カーブ"))
    
    def create_wb_tab(self):
        """ホワイトバランスタブ作成"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel(self.tr("ホワイトバランスを調整します")))
        
        # 色温度
        layout.addWidget(QLabel(self.tr("色温度:")))
        self.wb_temperature_slider = SliderWithSpinBox(-100, 100, 1, 0, 0)
        self.wb_temperature_slider.valueChanged.connect(self.on_wb_temperature_changed)
        layout.addWidget(self.wb_temperature_slider)
        
        # 色かぶり補正
        layout.addWidget(QLabel(self.tr("色かぶり補正:")))
        self.wb_tint_slider = SliderWithSpinBox(-100, 100, 1, 0, 0)
        self.wb_tint_slider.valueChanged.connect(self.on_wb_tint_changed)
        layout.addWidget(self.wb_tint_slider)
        
        layout.addStretch()
        
        # リセットボタン
        reset_btn = QPushButton(self.tr("このタブをリセット"))
        reset_btn.clicked.connect(self.reset_wb_tab)
        layout.addWidget(reset_btn)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, self.tr("WB"))
    
    def create_mask_tab(self):
        """マスクタブ作成"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # マスク作成
        create_layout = QHBoxLayout()
        self.ai_mask_btn = QPushButton(self.tr("AIマスク作成"))
        self.ai_mask_btn.clicked.connect(self.start_ai_mask_creation)
        self.cancel_mask_btn = QPushButton(self.tr("キャンセル"))
        self.cancel_mask_btn.clicked.connect(self.cancel_ai_mask_creation)
        self.cancel_mask_btn.setEnabled(False)
        
        create_layout.addWidget(self.ai_mask_btn)
        create_layout.addWidget(self.cancel_mask_btn)
        layout.addLayout(create_layout)
        
        # マスク選択
        select_layout = QHBoxLayout()
        self.mask_combo = QComboBox()
        self.mask_combo.addItem(self.tr("マスクなし"))
        self.mask_combo.currentTextChanged.connect(self.on_mask_selection_changed)
        
        self.delete_mask_btn = QPushButton(self.tr("マスクを削除"))
        self.delete_mask_btn.clicked.connect(self.delete_current_mask)
        
        select_layout.addWidget(self.mask_combo)
        select_layout.addWidget(self.delete_mask_btn)
        layout.addLayout(select_layout)
        
        # マスクを表示
        self.mask_display_checkbox = QCheckBox(self.tr("マスクを表示"))
        self.mask_display_checkbox.toggled.connect(self.on_mask_display_toggled)
        layout.addWidget(self.mask_display_checkbox)
        
        # マスクを反転して新規作成
        self.invert_mask_btn = QPushButton(self.tr("マスクを反転して新規作成"))
        self.invert_mask_btn.clicked.connect(self.invert_current_mask)
        layout.addWidget(self.invert_mask_btn)
        
        # マスク範囲の微調整
        layout.addWidget(QLabel(self.tr("マスク範囲の微調整:")))
        self.mask_range_slider = SliderWithSpinBox(-4.0, 4.0, 0.02, 2, 0.0)
        self.mask_range_slider.valueChanged.connect(self.on_mask_range_changed)
        layout.addWidget(self.mask_range_slider)
        
        layout.addStretch()
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, self.tr("マスク"))
    
    def create_metadata_tab(self):
        """メタデータタブ作成"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel(self.tr("画像のメタデータ情報")))
        
        # テーブルウィジェット
        self.metadata_table = QTableWidget()
        self.metadata_table.setColumnCount(2)
        self.metadata_table.setHorizontalHeaderLabels([self.tr("キー"), self.tr("値")])
        
        # テーブル設定
        self.metadata_table.setWordWrap(False)
        self.metadata_table.setTextElideMode(Qt.ElideNone)
        self.metadata_table.verticalHeader().setVisible(False)
        self.metadata_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.metadata_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)

        layout.addWidget(self.metadata_table)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, self.tr("メタデータ"))
    
    def open_file(self):
        """ファイルを開く"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("写真を開く"),
            "",
            f"{self.tr('画像ファイル')} (*.raw *.cr2 *.cr3 *.nef *.arw *.dng *.jpg *.jpeg *.png *.tiff *.tif);;{self.tr('すべてのファイル')} (*)"
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """画像を読み込み"""
        if not RAW_EDITOR_AVAILABLE:
            QMessageBox.critical(self, self.tr("エラー"), self.tr("raw_image_editor が利用できません"))
            return
        
        # プログレスダイアログ表示
        progress = QProgressDialog(self.tr("画像を読み込み中..."), None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        # ワーカースレッドで読み込み
        self.load_worker = ImageLoadWorker(file_path)
        self.load_worker.finished.connect(self.on_image_loaded)
        self.load_worker.error.connect(self.on_load_error)
        self.load_worker.finished.connect(lambda: progress.close())
        self.load_worker.error.connect(lambda: progress.close())
        self.load_worker.start()
    
    def on_image_loaded(self, editor, medium_editor, small_editor):
        """画像読み込み完了"""
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
        self.mask_combo.blockSignals(True)
        self.mask_combo.clear()
        self.mask_combo.addItem(self.tr("マスクなし"))
        self.mask_combo.blockSignals(False)

        self.update_ui_from_parameters()
        
        # 画像表示を更新
        self.update_image_display()
        
        # メタデータ表示を更新
        self.update_metadata_display()
    
    def on_load_error(self, error_message):
        """画像読み込みエラー"""
        QMessageBox.critical(self, self.tr("エラー"), error_message)
    
    def reset_ui_controls(self):
        """UI制御をリセット"""
        # このメソッドは update_ui_from_parameters に置き換えられました
        # 古い呼び出し元が残っている場合があるため、念のため残します
        self.update_ui_from_parameters()
    
        def update_image_display(self, force_full=False):
            """画像表示を更新"""
            if self.editor is None:
                return
            
            # 表示するエディターを選択
            if self.dragging and not force_full:
                display_editor = self.small_editor
            else:
                display_editor = self.medium_editor
            
            try:
                # 1. Reset and rebuild editor state from GUI state
                self.apply_parameters_to_editor(display_editor)
                
                # 2. Apply adjustments
                display_editor.apply_adjustments()
                
                # 3. Get the resulting image
                image_array = display_editor.as_uint8()
    
                # 4. If mask overlay is enabled, create and blend it
                if self.mask_display_enabled and self.current_mask_name and self.current_mask_name != self.tr("マスクなし"):
                    if self.current_mask_name in display_editor.mask_adjustments_values:
                        mask_data = display_editor.mask_adjustments_values[self.current_mask_name]
                        mask_range_value = self.edit_params.mask_range
                        boolean_mask = mask_range_value < mask_data["ndarray"]
                        mask_image_pil = display_editor.get_mask_image(boolean_mask)
                        
                        # Convert PIL overlay to OpenCV format
                        overlay_array = np.array(mask_image_pil)
                        if overlay_array.shape[2] == 4: # RGBA
                            overlay_array = cv2.cvtColor(overlay_array, cv2.COLOR_RGBA2RGB)
                        
                        # Ensure overlay has same size as image
                        if overlay_array.shape[:2] != image_array.shape[:2]:
                            overlay_array = cv2.resize(overlay_array, (image_array.shape[1], image_array.shape[0]))
    
                        # Blend the image and the overlay
                        blended_image = cv2.addWeighted(image_array, 0.7, overlay_array, 0.3, 0)
                        
                        self.image_widget.set_image(blended_image, is_mask_overlay=True)
                    else:
                        # Mask not found, show normal image
                        self.image_widget.set_image(image_array)
                else:
                    # Mask display not enabled, show normal image
                    self.image_widget.set_image(image_array)
                    
            except Exception as e:
                print(f"画像表示の更新でエラー: {e}")
                traceback.print_exc()   


    def update_image_display(self, force_full=False):
        """画像表示を更新"""
        if self.editor is None:
            return
        
        # 表示するエディターを選択
        if self.dragging and not force_full:
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
            if image_array is not None:
                hist_data = {}
                # グレースケールヒストグラム（白）
                gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                hist_data['white'] = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
                
                # RGBヒストグラム
                colors = ('r', 'g', 'b')
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([image_array], [i], None, [256], [0, 256])
                    hist_data[color] = hist
                
                # brightness_curveが初期化されているか確認
                if hasattr(self, 'brightness_curve'):
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
        """編集パラメータをエディターに適用（デバッグ版）"""
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
                hue_curve = self._get_curve_array_from_points(params.hue_curve_points)
                editor.set_hls_hue_tone_curve(curve=hue_curve, mask_name=None)
                saturation_curve = self._get_curve_array_from_points(params.saturation_curve_points)
                editor.set_hls_saturation_tone_curve(curve=(saturation_curve * 2), mask_name=None)
                lightness_curve = self._get_curve_array_from_points(params.lightness_curve_points)
                editor.set_hls_lightness_tone_curve(curve=(lightness_curve * 2), mask_name=None)

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
                hue_curve = self._get_curve_array_from_points(params.hue_curve_points)
                editor.set_hls_hue_tone_curve(curve=hue_curve, mask_name=mask_name)
                saturation_curve = self._get_curve_array_from_points(params.saturation_curve_points)
                editor.set_hls_saturation_tone_curve(curve=(saturation_curve * 2), mask_name=mask_name)
                lightness_curve = self._get_curve_array_from_points(params.lightness_curve_points)
                editor.set_hls_lightness_tone_curve(curve=(lightness_curve * 2), mask_name=mask_name)
                
                # マスク範囲を更新
                if mask_name in editor.mask_adjustments_values:
                    editor.mask_adjustments_values[mask_name]["mask_range_value"] = params.mask_range
                
        except Exception as e:
            print(f"パラメータ適用でエラー: {e}")
            traceback.print_exc()
    
    def update_metadata_display(self):
        """メタデータ表示を更新"""
        if self.editor is None or self.editor.metadata is None:
            return
        
        try:
            # 日本語メタデータを取得
            japanese_metadata = self.editor.metadata.display_japanese("dict")
            
            self.metadata_table.setRowCount(len(japanese_metadata))

            for row, (key, value) in enumerate(japanese_metadata.items()):
                key_item = QTableWidgetItem(str(key))
                value_item = QTableWidgetItem(str(value))
                
                self.metadata_table.setItem(row, 0, key_item)
                self.metadata_table.setItem(row, 1, value_item)
            
            self.metadata_table.resizeColumnsToContents()
            
        except Exception as e:
            print(f"メタデータ表示でエラー: {e}")
    
    # 露出パラメータ変更ハンドラー
    def on_exposure_changed(self, value):
        self.edit_params.exposure = value
        self.start_drag_timer()
    
    def on_contrast_changed(self, value):
        self.edit_params.contrast = int(value)
        self.start_drag_timer()
    
    def on_shadow_changed(self, value):
        self.edit_params.shadow = int(value)
        self.start_drag_timer()
    
    def on_highlight_changed(self, value):
        self.edit_params.highlight = int(value)
        self.start_drag_timer()
    
    def on_black_changed(self, value):
        self.edit_params.black = int(value)
        self.start_drag_timer()
    
    def on_white_changed(self, value):
        self.edit_params.white = int(value)
        self.start_drag_timer()
    
    # ホワイトバランス変更ハンドラー
    def on_wb_temperature_changed(self, value):
        self.edit_params.wb_temperature = int(value)
        self.start_drag_timer()
    
    def on_wb_tint_changed(self, value):
        self.edit_params.wb_tint = int(value)
        self.start_drag_timer()
    
    # トーンカーブ変更ハンドラー
    def on_brightness_curve_changed(self, points):
        self.edit_params.brightness_curve_points = points
        self.start_drag_timer()
    
    def on_hue_curve_changed(self, points):
        self.edit_params.hue_curve_points = points
        self.start_drag_timer()
    
    def on_saturation_curve_changed(self, points):
        self.edit_params.saturation_curve_points = points
        self.start_drag_timer()
    
    def on_lightness_curve_changed(self, points):
        self.edit_params.lightness_curve_points = points
        self.start_drag_timer()
    
    def on_mask_range_changed(self, value):
        """マスク範囲の微調整"""
        self.edit_params.mask_range = value
        self.start_drag_timer()
    
    def start_drag_timer(self):
        """ドラッグタイマーを開始"""
        self.dragging = True
        self.drag_timer.stop()
        self.drag_timer.start(100)  # 100ms後にドラッグ終了と判定
        self.update_image_display()
    
    def on_drag_timeout(self):
        """ドラッグタイムアウト（高品質表示に切り替え）"""
        self.dragging = False
        self.update_image_display(force_full=True)
    
    def reset_exposure_tab(self):
        """露出タブリセット"""
        self.edit_params.exposure = 0.0
        self.edit_params.contrast = 0
        self.edit_params.shadow = 0
        self.edit_params.highlight = 0
        self.edit_params.black = 0
        self.edit_params.white = 0
        
        self.exposure_slider.setValue(0.0)
        self.contrast_slider.setValue(0)
        self.shadow_slider.setValue(0)
        self.highlight_slider.setValue(0)
        self.black_slider.setValue(0)
        self.white_slider.setValue(0)
        
        self.update_image_display()
    
    def reset_wb_tab(self):
        """ホワイトバランスタブリセット"""
        self.edit_params.wb_temperature = 0
        self.edit_params.wb_tint = 0
        
        self.wb_temperature_slider.setValue(0)
        self.wb_tint_slider.setValue(0)
        
        self.update_image_display()
    
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
            QMessageBox.warning(self, self.tr("警告"), self.tr("エクスポートする画像がありません"))
            return
        
        # エクスポート設定ダイアログ
        dialog = ExportDialog(self, self)
        if dialog.exec() != QDialog.Accepted:
            return
        
        # ファイル名を生成
        if self.current_file_path:
            base_name = Path(self.current_file_path).stem
        else:
            base_name = "exported"
        
        format_str = dialog.get_format().lower()
        
        suggested_name = f"{base_name}_edited.{format_str}"
        
        # 保存先ダイアログ
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("画像をエクスポート"),
            suggested_name,
            f"{dialog.get_format()} (*.{format_str});;{self.tr('すべてのファイル')} (*)"
        )
        
        if not file_path:
            return
        
        # プログレスダイアログ
        progress = QProgressDialog(self.tr("画像をエクスポート中..."), None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        # エディターのディープコピーを作成してエクスポート用に使用
        try:
            # エディターの現在の状態を完全にリセット
            self.editor.reset()
            
            # 最終的なパラメータを適用
            self.apply_parameters_to_editor(self.editor)
            self.editor.apply_adjustments()

            # ワーカースレッドでエクスポート
            self.export_worker = ExportWorker(self.editor, file_path, dialog.get_quality())
            self.export_worker.finished.connect(self.on_export_finished)
            self.export_worker.error.connect(self.on_export_error)
            self.export_worker.finished.connect(lambda: progress.close())
            self.export_worker.error.connect(lambda: progress.close())
            
            self.export_worker.start()
            
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, self.tr("エラー"), f"{self.tr('エクスポートの準備中にエラーが発生しました')}: {str(e)}")
            print(f"Export preparation error: {e}")
            traceback.print_exc()
    
    def on_export_finished(self, export_path):
        """エクスポート完了"""
        QMessageBox.information(
            self,
            self.tr("エクスポート完了"),
            f"{self.tr('画像を保存しました')}:{export_path}"
        )
    
    def on_export_error(self, error_message):
        """エクスポートエラー"""
        QMessageBox.critical(self, self.tr("エラー"), error_message)
    
    def save_preset(self):
        """プリセット保存"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("プリセットを保存"),
            "",
            f"{self.tr('プリセットファイル')} (*.json);;{self.tr('すべてのファイル')} (*)"
        )
        
        if not file_path:
            return
        
        try:
            # 「マスクなし」の編集パラメータを辞書に変換
            preset_data = asdict(self.mask_edit_params[self.tr("マスクなし")])
            
            # JSONファイルに保存
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)
            
            QMessageBox.information(self, self.tr("情報"), self.tr("プリセットを保存しました"))
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("エラー"), f"{self.tr('プリセットの保存に失敗しました')}: {str(e)}")
    
    def load_preset(self):
        """プリセット読み込み"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("プリセットを読み込み"),
            "",
            f"{self.tr('プリセットファイル')} (*.json);;{self.tr('すべてのファイル')} (*)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                preset_data = json.load(f)
            
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
            
            QMessageBox.information(self, self.tr("情報"), self.tr("プリセットを読み込みました"))
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("エラー"), f"{self.tr('プリセットの読み込みに失敗しました')}: {str(e)}")
    
    def update_ui_from_parameters(self):
        """パラメータからUIを更新"""
        # 露出パラメータ
        self.exposure_slider.setValue(self.edit_params.exposure)
        self.contrast_slider.setValue(self.edit_params.contrast)
        self.shadow_slider.setValue(self.edit_params.shadow)
        self.highlight_slider.setValue(self.edit_params.highlight)
        self.black_slider.setValue(self.edit_params.black)
        self.white_slider.setValue(self.edit_params.white)
        
        # ホワイトバランス
        self.wb_temperature_slider.setValue(self.edit_params.wb_temperature)
        self.wb_tint_slider.setValue(self.edit_params.wb_tint)
        
        # トーンカーブ
        self.brightness_curve.set_control_points(self.edit_params.brightness_curve_points)
        self.hue_curve.set_control_points(self.edit_params.hue_curve_points)
        self.saturation_curve.set_control_points(self.edit_params.saturation_curve_points)
        self.lightness_curve.set_control_points(self.edit_params.lightness_curve_points)

        # マスク範囲
        self.mask_range_slider.setValue(self.edit_params.mask_range)
    
    def show_settings(self):
        """設定ダイアログを表示"""
        dialog = SettingsDialog(self.settings_manager, self, self)
        dialog.exec()
    
    # マスク関連メソッド
    def start_ai_mask_creation(self):
        """AIマスク作成開始"""
        if self.editor is None:
            QMessageBox.warning(self, self.tr("警告"), self.tr("画像が読み込まれていません"))
            return
        
        self.image_widget.set_ai_mask_mode(True)
        self.ai_mask_btn.setEnabled(False)
        self.cancel_mask_btn.setEnabled(True)
    
    def cancel_ai_mask_creation(self):
        """AIマスク作成キャンセル"""
        self.image_widget.set_ai_mask_mode(False)
        self.ai_mask_btn.setEnabled(True)
        self.cancel_mask_btn.setEnabled(False)
    
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
        progress = QProgressDialog(self.tr("AIマスク作成中..."), None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        # AIマスクを生成する前に、すべての編集をフル解像度のエディターに適用する
        try:
            self.editor.reset()
            self.apply_parameters_to_editor(self.editor)
            self.editor.apply_adjustments()
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, self.tr("エラー"), f"{self.tr('AIマスクの準備中にエラーが発生しました')}: {str(e)}")
            print(f"AI mask preparation error: {e}")
            traceback.print_exc()
            return

        # ワーカースレッドでマスク作成
        self.ai_mask_worker = AIMaskWorker(self.editor, x, y, mask_name)
        self.ai_mask_worker.finished.connect(self.on_ai_mask_created)
        self.ai_mask_worker.error.connect(self.on_ai_mask_error)
        self.ai_mask_worker.finished.connect(lambda: progress.close())
        self.ai_mask_worker.error.connect(lambda: progress.close())
        self.ai_mask_worker.start()
    
    def on_ai_mask_created(self, mask_name):
        """AIマスク作成完了"""
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
        
        # コンボボックスにマスクを追加し、それを選択状態にする
        self.mask_combo.addItem(mask_name)
        self.mask_combo.setCurrentText(mask_name)
        
        QMessageBox.information(
            self,
            self.tr("AIマスク作成完了"),
            f"{self.tr('AIマスクを作成しました')}:{mask_name}"
        )
    
    def on_ai_mask_error(self, error_message):
        """AIマスク作成エラー"""
        QMessageBox.critical(self, self.tr("エラー"), error_message)
    
    def on_mask_selection_changed(self, mask_name):
        """マスク選択変更"""
        if not mask_name:  # コンボボックスがクリアされた場合など
            return

        self.current_mask_name = mask_name
        
        # UIを新しいマスクのパラメータで更新
        self.update_ui_from_parameters()
        self.update_image_display()
    
    def delete_current_mask(self):
        """現在のマスクを削除"""
        if self.current_mask_name == self.tr("マスクなし"):
            return
        
        # 確認ダイアログ
        reply = QMessageBox.question(
            self,
            self.tr("確認"),
            f"{self.tr('マスク')} '{self.current_mask_name}' {self.tr('を削除しますか？')}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
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
        index = self.mask_combo.findText(mask_to_delete)
        if index >= 0:
            self.mask_combo.removeItem(index)
        
        # "マスクなし" を選択状態にする (on_mask_selection_changedがトリガーされる)
        self.mask_combo.setCurrentText(self.tr("マスクなし"))
    
    def invert_current_mask(self):
        """現在のマスクを反転して新規作成"""
        if self.current_mask_name is None or self.current_mask_name == self.tr("マスクなし"):
            QMessageBox.warning(self, self.tr("警告"), self.tr("反転するマスクが選択されていません"))
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
        self.mask_combo.addItem(new_mask_name)
        self.mask_combo.setCurrentText(new_mask_name)
        
        QMessageBox.information(
            self,
            self.tr("マスク作成完了"),
            f"{self.tr('反転マスクを作成しました')}:{new_mask_name}"
        )
    
    def on_mask_display_toggled(self, checked):
        """マスク表示切り替え"""
        self.mask_display_enabled = checked
        self.update_image_display()

def create_language_file_if_not_exists(path: Path):
    """言語ファイルが存在しない場合に作成する"""
    if path.exists():
        return
    
    languages = {
        "English": {
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
            "シャドウ:": "Shadow:",
            "ハイライト:": "Highlight:",
            "黒レベル:": "Black Level:",
            "白レベル:": "White Level:",
            "このタブをリセット": "Reset This Tab",
            "明るさカーブ": "Brightness Curve",
            "明るさのトーンカーブを調整します": "Adjust the brightness tone curve.",
            "トーンカーブをリセット": "Reset Tone Curve",
            "色相カーブ": "Hue Curve",
            "色相のトーンカーブを調整します": "Adjust the hue tone curve.",
            "彩度カーブ": "Saturation Curve",
            "彩度のトーンカーブを調整します": "Adjust the saturation tone curve.",
            "輝度カーブ": "Lightness Curve",
            "輝度のトーンカーブを調整します": "Adjust the lightness tone curve.",
            "WB": "WB",
            "ホワイトバランスを調整します": "Adjust white balance.",
            "色温度:": "Temperature:",
            "色かぶり補正:": "Tint:",
            "マスク": "Mask",
            "AIマスク作成": "Create AI Mask",
            "キャンセル": "Cancel",
            "マスクを選択": "Select Mask",
            "マスクを削除": "Delete Mask",
            "マスクなし": "No Mask",
            "マスクを表示": "Show Mask",
            "マスクを反転して新規作成": "Invert Mask and Create New",
            "マスク範囲の微調整:": "Fine-tune Mask Range:",
            "メタデータ": "Metadata",
            "画像のメタデータ情報": "Image metadata information.",
            "キー": "Key",
            "値": "Value",
            "処理デバイス": "Processing Device",
            "言語": "Language",
            "言語:": "Language",
            "処理デバイス:": "Processing Device:",
            "設定を保存": "Save Settings",
            "画像をドラッグ&ドロップまたは「ファイル」→「写真を開く」": "Drag & Drop an image or go to File -> Open Photo",
            "エクスポート設定": "Export Settings",
            "画質:": "Quality:",
            "形式:": "Format:",
            "エクスポート": "Export",
            "情報": "Information",
            "設定を保存しました。ソフトウェアを再起動してください。": "Settings saved.Please restart the software.",
            "警告": "Warning",
            "設定の保存に失敗しました": "Failed to save settings",
            "画像ファイル": "Image Files",
            "すべてのファイル": "All Files",
            "エラー": "Error",
            "raw_image_editor が利用できません": "raw_image_editor is not available",
            "画像を読み込み中...": "Loading image...",
            "エクスポートする画像がありません": "No image to export",
            "画像をエクスポート": "Export Image",
            "画像をエクスポート中...": "Exporting image...",
            "エクスポートの準備中にエラーが発生しました": "Error during export preparation",
            "エクスポート完了": "Export Complete",
            "画像を保存しました": "Image saved",
            "プリセットを保存": "Save Preset",
            "プリセットファイル": "Preset Files",
            "プリセットを保存しました": "Preset saved",
            "プリセットの保存に失敗しました": "Failed to save preset",
            "プリセットを読み込み": "Load Preset",
            "プリセットを読み込みました": "Preset loaded",
            "プリセットの読み込みに失敗しました": "Failed to load preset",
            "画像が読み込まれていません": "No image loaded",
            "AIマスク作成中...": "Creating AI mask...",
            "AIマスクを作成しました": "AI mask created",
            "AIマスク作成完了": "AI Mask Creation Complete",
            "確認": "Confirm",
            "を削除しますか？": " will be deleted. Are you sure?",
            "反転するマスクが選択されていません": "No mask selected to invert",
            "反転マスクを作成しました": "Inverted mask created",
            "マスク作成完了": "Mask Creation Complete"
        },
        "日本語": {
            "RAW現像ソフト": "RAW現像ソフト",
            "ファイル": "ファイル",
            "写真を開く": "写真を開く",
            "画像をエクスポート": "画像をエクスポート",
            "編集": "編集",
            "すべての編集をリセット": "すべての編集をリセット",
            "プリセット": "プリセット",
            "現在の編集をプリセットとして保存": "現在の編集をプリセットとして保存",
            "プリセットを読み込み": "プリセットを読み込み",
            "設定": "設定",
            "露出": "露出",
            "基本的な露出調整を行います": "基本的な露出調整を行います",
            "コントラスト:": "コントラスト:",
            "シャドウ:": "シャドウ:",
            "ハイライト:": "ハイライト:",
            "黒レベル:": "黒レベル:",
            "白レベル:": "白レベル:",
            "このタブをリセット": "このタブをリセット",
            "明るさカーブ": "明るさカーブ",
            "明るさのトーンカーブを調整します": "明るさのトーンカーブを調整します",
            "トーンカーブをリセット": "トーンカーブをリセット",
            "色相カーブ": "色相カーブ",
            "色相のトーンカーブを調整します": "色相のトーンカーブを調整します",
            "彩度カーブ": "彩度カーブ",
            "彩度のトーンカーブを調整します": "彩度のトーンカーブを調整します",
            "輝度カーブ": "輝度カーブ",
            "輝度のトーンカーブを調整します": "輝度のトーンカーブを調整します",
            "WB": "WB",
            "ホワイトバランスを調整します": "ホワイトバランスを調整します",
            "色温度:": "色温度:",
            "色かぶり補正:": "色かぶり補正:",
            "マスク": "マスク",
            "AIマスク作成": "AIマスク作成",
            "キャンセル": "キャンセル",
            "マスクを選択": "マスクを選択",
            "マスクを削除": "マスクを削除",
            "マスクなし": "マスクなし",
            "マスクを表示": "マスクを表示",
            "マスクを反転して新規作成": "マスクを反転して新規作成",
            "マスク範囲の微調整:": "マスク範囲の微調整:",
            "メタデータ": "メタデータ",
            "画像のメタデータ情報": "画像のメタデータ情報",
            "キー": "キー",
            "値": "値",
            "処理デバイス": "処理デバイス",
            "言語": "言語",
            "設定を保存": "設定を保存",
            "画像をドラッグ&ドロップまたは「ファイル」→「写真を開く」": "画像をドラッグ&ドロップまたは「ファイル」→「写真を開く」",
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
            "画像を読み込み中...": "画像を読み込み中...",
            "エクスポートする画像がありません": "エクスポートする画像がありません",
            "画像をエクスポート": "画像をエクスポート",
            "画像をエクスポート中...": "画像をエクスポート中...",
            "エクスポートの準備中にエラーが発生しました": "エクスポートの準備中にエラーが発生しました",
            "エクスポート完了": "エクスポート完了",
            "画像を保存しました": "画像を保存しました",
            "プリセットを保存": "プリセットを保存",
            "プリセットファイル": "プリセットファイル",
            "プリセットを保存しました": "プリセットを保存しました",
            "プリセットの保存に失敗しました": "プリセットの保存に失敗しました",
            "プリセットを読み込み": "プリセットを読み込み",
            "プリセットを読み込みました": "プリセットを読み込みました",
            "プリセットの読み込みに失敗しました": "プリセットの読み込みに失敗しました",
            "画像が読み込まれていません": "画像が読み込まれていません",
            "AIマスク作成中...": "AIマスク作成中...",
            "AIマスクを作成しました": "AIマスクを作成しました",
            "AIマスク作成完了": "AIマスク作成完了",
            "確認": "確認",
            "を削除しますか？": "を削除しますか？",
            "反転するマスクが選択されていません": "反転するマスクが選択されていません",
            "反転マスクを作成しました": "反転マスクを作成しました",
            "マスク作成完了": "マスク作成完了"
        }
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(languages, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"言語ファイルの作成に失敗しました: {e}")

def main():
    """メイン関数"""
    app = QApplication(sys.argv)
    
    # アプリケーション情報設定
    app.setApplicationName("RAW現像ソフト")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("PhotoEditor")

    # 初回起動時に言語ファイルを作成
    sm = SettingsManager()
    sm.__post_init__()
    create_language_file_if_not_exists(sm.language_path)
    
    # メインウィンドウ作成
    window = RAWDevelopmentGUI()
    window.show()
    
    # コマンドライン引数でファイルが渡された場合、UI表示後に読み込む
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            # UIのイベントループが開始してから実行されるようにタイマーを使う
            QTimer.singleShot(100, lambda: window.load_image(file_path))
    
    # イベントループ開始
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
