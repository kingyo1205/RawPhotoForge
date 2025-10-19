# __init__.py

import os
import wgpu
from typing import Optional, Type, Union, List, Dict, Tuple, Any
from pathlib import Path
import numpy as np

import photo_metadata

WGPU_AVAILABLE = False
try:
    import wgpu
    import wgpu.backends.wgpu_native
    WGPU_AVAILABLE = True
except Exception:
    raise RuntimeError("wgpu is not available")



DEVICE: str = ''
is_initialized = False
    
from .editor import RAWImageEditor

def get_wgpu_adapters() -> List[wgpu.GPUAdapter]:
    """
    wgpu で利用可能なアダプターを返す。
    """
    if not WGPU_AVAILABLE:
        return []
    
    adapters = wgpu.gpu.enumerate_adapters_sync()
    
    
    return adapters




def get_device() -> str :
    """
    現在のデバイスを返す
    """
    return DEVICE


    


def init(device_index: int = 0) -> None:
    """
    raw_image_editorを初期化します。

    Parameters
    ----------
    device_index : int, optional
        使用するwgpuデバイスのインデックスを指定します。デフォルトは 0 です。
    """
    
    global DEVICE
    global RAWImageEditor
    global is_initialized

    if is_initialized:
        raise RuntimeError("Already initialized")

    if not WGPU_AVAILABLE:
        raise ImportError("wgpu is not installed. Please install it to use the wgpu backend.")
    
    try:
        adapters = get_wgpu_adapters()
        print([a.summary for a in adapters])
        
        
        adapter = adapters[device_index]
        device = adapter.request_device_sync()

    except IndexError:
        raise ValueError(f"wgpu device index {device_index} is out of range.")


    with open(os.path.join(os.path.dirname(__file__), "wgpu_kernel.wgsl"), "r", encoding="utf-8") as f:
        wgpu_kernel_code = f.read()
    
    RAWImageEditor.device = device
    
    wgpu_kernel_program = device.create_shader_module(code=wgpu_kernel_code)

    # Create bind group layouts for each pipeline
    simple_layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])

    tone_curve_lut_layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])

    tone_curve_by_hue_layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 6, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])

    white_balance_layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 7, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])

    vignette_layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 8, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])

    # Create pipelines with the correct layouts
    RAWImageEditor.clip_0_1_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[simple_layout]),
        compute={"module": wgpu_kernel_program, "entry_point": "clip_0_1"},
    )
    RAWImageEditor.to_linear_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[simple_layout]),
        compute={"module": wgpu_kernel_program, "entry_point": "to_linear"},
    )
    RAWImageEditor.to_srgb_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[simple_layout]),
        compute={"module": wgpu_kernel_program, "entry_point": "to_srgb"},
    )
    RAWImageEditor.rgb_to_hls_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[simple_layout]),
        compute={"module": wgpu_kernel_program, "entry_point": "rgb_to_hls"},
    )
    RAWImageEditor.hls_to_rgb_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[simple_layout]),
        compute={"module": wgpu_kernel_program, "entry_point": "hls_to_rgb"},
    )

    RAWImageEditor.tone_curve_lut_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[tone_curve_lut_layout]),
        compute={"module": wgpu_kernel_program, "entry_point": "tone_curve_lut"},
    )
    RAWImageEditor.tone_curve_by_hue_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[tone_curve_by_hue_layout]),
        compute={"module": wgpu_kernel_program, "entry_point": "tone_curve_by_hue"},
    )
    RAWImageEditor.white_balance_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[white_balance_layout]),
        compute={"module": wgpu_kernel_program, "entry_point": "white_balance"},
    )
    RAWImageEditor.vignette_effect_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[vignette_layout]),
        compute={"module": wgpu_kernel_program, "entry_point": "vignette_effect"},
    )


    DEVICE = f'wgpu:{device_index}'
    is_initialized = True

    backend = device.adapter_info.get("backend_type")
    print(f"RAWImageEditor initialized on {DEVICE}, {device.adapter_info.device}, {backend}")

def get_is_initialized() -> bool:
    """RAWImageEditorが初期化されているかどうかを返します。"""
    return is_initialized
