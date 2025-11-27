# __init__.py


from typing import Optional, Type, Union, List, Dict, Tuple, Any
from pathlib import Path
import sys
import numpy as np
import photo_metadata

SLANGPY_AVAILABLE = False
try:
    import slangpy
    SLANGPY_AVAILABLE = True
except ImportError:
    raise RuntimeError("slangpy is not available")

import slangpy as spy

DEVICE: str = ''
is_initialized = False

from .editor import RAWImageEditor


if sys.platform.startswith("win"):
    device_type = spy.DeviceType.vulkan
elif sys.platform.startswith("linux"):
    device_type = spy.DeviceType.vulkan
elif sys.platform == "darwin":
    device_type = spy.DeviceType.metal
else:
    device_type = spy.DeviceType.automatic


def get_slangpy_adapters() -> List[spy.AdapterInfo]:
    """
    slangpy で利用可能なアダプター名を返す。
    """
    if not SLANGPY_AVAILABLE:
        return []
    
    # slangpy doesn't have a direct equivalent to enumerate_adapters_sync to get names beforehand.
    # We can, however, list devices if we create a device. For now, we'll return a conceptual list.
    # This function can be expanded if slangpy provides more introspection capabilities.
    try:
        # Creating a default device to check availability.
        device = spy.create_device(type=device_type)
        return device.enumerate_adapters()
    except Exception as e:
        print(f"Could not get Slangpy adapter info: {e}")
        return []

def get_device() -> str:
    """
    現在のデバイス情報を返す
    """
    return DEVICE

def init(device_index: int = 0) -> None:
    """
    raw_image_editorを初期化します。
    """
    global DEVICE
    global is_initialized

    if is_initialized:
        print("Already initialized.")
        return

    if not SLANGPY_AVAILABLE:
        raise ImportError("slangpy is not installed. Please install it to use the slangpy backend.")
    
    try:
        
        device = spy.create_device(type=device_type, adapter_luid=get_slangpy_adapters()[device_index].luid)

        RAWImageEditor.device = device

        shader_path = Path(__file__).parent / "slang_kernel.slang"
        entry_points = [
            "to_linear", "clip_0_1", "to_srgb", "tone_curve_lut",
            "tone_curve_by_hue", "rgb_to_hls", "hls_to_rgb",
            "white_balance", "vignette_effect"
        ]
        for name in entry_points:
            program = RAWImageEditor.device.load_program(
                module_name=str(shader_path),
                entry_point_names=[name]
            )
            RAWImageEditor.kernels[name] = RAWImageEditor.device.create_compute_kernel(
                program
            )

        adapter_info = device.info.adapter_name
        DEVICE = f"slangpy:{device_index}"
        is_initialized = True
        print(f"RAWImageEditor (slangpy backend) initialized. Device: {adapter_info}, Backend: {device.info.api_name}")

    except Exception as e:
        raise RuntimeError(f"Failed to initialize slangpy device: {e}")

def get_is_initialized() -> bool:
    """RAWImageEditorが初期化されているかどうかを返します。"""
    return is_initialized

