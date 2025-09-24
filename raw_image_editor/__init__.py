# __init__.py

import photo_metadata
from typing import Optional, Type, Union, List, Dict, Tuple, Any
import os
from pathlib import Path

try:
    
    import pyopencl as cl
    platforms = cl.get_platforms()
    OPENCL_AVAILABLE = len(platforms) > 0
except Exception:
    OPENCL_AVAILABLE = False


DEVICE: str = 'cpu'
is_initialized = False
    
from .numpy_backend import RAWImageEditorNumpy







RAWImageEditor: Type[RAWImageEditorNumpy] = None


def get_opencl_ctxs() -> List["cl.Context"] :
    """
    OpenCL で利用可能な Context を返す。
    OpenCL が利用可能でない場合は空のリストを返す。
    """
    if not OPENCL_AVAILABLE:
        return []
    ctxs = []
    for pf in cl.get_platforms():
        for dev in pf.get_devices(device_type=cl.device_type.ALL):
            ctxs.append(cl.Context([dev]))
    return ctxs

def get_device() -> str :
    """
    現在のデバイスを返す
    """
    return DEVICE


    


def init(device: str = 'cpu'):
    """
    raw_image_editorを初期化します。

    Parameters
    ----------
    device : str, optional
        使用するバックエンドを指定します。デフォルトは 'cpu' です。
        'cpu': NumPyを使用したCPUバックエンド。
        'opencl:{index}': OpenCLを使用したGPUバックエンド。
                           indexにはget_opencl_ctxs()で得られるデバイスのインデックスを指定します。
    """
    global DEVICE
    global RAWImageEditor
    global is_initialized

    if is_initialized:
        raise RuntimeError("Already initialized")

    device = device.lower()
    
    
    if device == 'cpu':
        RAWImageEditor = RAWImageEditorNumpy


    elif device.startswith('opencl:'):
        if not OPENCL_AVAILABLE:
            raise ImportError("pyopencl is not installed or no OpenCL platforms found. Please install it to use the opencl backend.")
        
        parts = device.split(':')
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(f"Invalid opencl device format: '{device}'. Expected 'opencl:{{index}}'.")
        
        index = int(parts[1])
        
        import raw_image_editor.opencl_backend as opencl_backend

        try:
            opencl_backend.ctx = get_opencl_ctxs()[index]
        except IndexError:
            raise ValueError(f"OpenCL device index {index} is out of range.")


        with open(os.path.join(os.path.dirname(__file__), "opencl_kernel.cl"), "r", encoding="utf-8") as f:
            ocl_kernel_code = f.read()
        
        from .opencl_backend import RAWImageEditorOpenCL
        RAWImageEditor = RAWImageEditorOpenCL
        RAWImageEditor.ctx = opencl_backend.ctx
        RAWImageEditor.queue = opencl_backend.cl.CommandQueue(opencl_backend.ctx)
        ocl_kernel_program = cl.Program(opencl_backend.ctx, ocl_kernel_code).build()

        RAWImageEditor.clahe = opencl_backend.OpenCLCLAHE(opencl_backend.ctx, RAWImageEditor.queue)
        

        RAWImageEditor.clip_0_1_kernel = cl.Kernel(ocl_kernel_program, "clip_0_1")
        RAWImageEditor.tone_curve_lut_kernel = cl.Kernel(ocl_kernel_program, "tone_curve_lut")
        RAWImageEditor.rgb_to_hls_kernel = cl.Kernel(ocl_kernel_program, "rgb_to_hls")
        RAWImageEditor.hls_to_rgb_kernel = cl.Kernel(ocl_kernel_program, "hls_to_rgb")
        RAWImageEditor.to_linear_kernel = cl.Kernel(ocl_kernel_program, "to_linear")
        RAWImageEditor.to_srgb_kernel = cl.Kernel(ocl_kernel_program, "to_srgb")
        RAWImageEditor.tone_curve_by_hue_kernel = cl.Kernel(ocl_kernel_program, "tone_curve_by_hue")
        RAWImageEditor.white_balance_kernel = cl.Kernel(ocl_kernel_program, "white_balance")
    else:
        raise ValueError(f"Invalid device: '{device}'. Supported devices are 'cpu', 'opencl:{{index}}'.")
    
    DEVICE = device
    is_initialized = True


def get_is_initialized():
    """RAWImageEditorが初期化されているかどうかを返します。"""
    return is_initialized
    
    





