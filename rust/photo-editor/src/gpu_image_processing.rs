// gpu_image_processing.rs

use crate::errors::PhotoEditorError;
use crate::{EditParameters, GpuMask, Image};
use ndarray::Array1;
use std::borrow::Cow;
use std::sync::Arc;
use std::time;
use wgpu::util::DeviceExt;

use crate::interpolation;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BaseParams {
    pub width: u32,
    pub height: u32,
    pub num_masks: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuEditParameters {
    pub r_gain: f32,
    pub g_gain: f32,
    pub b_gain: f32,
    pub vignette: i32,
    pub lens_distortion: f32,
}

pub struct GpuProcessor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuProcessor {
    pub fn new(adapter_index: usize) -> Result<Self, PhotoEditorError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .into_iter()
            .nth(adapter_index)
            .ok_or_else(|| PhotoEditorError::gpu_initialization_no_source())?;

        // ここ重要: テクスチャ配列を使うので features 必須
        let required_features =
            wgpu::Features::TEXTURE_BINDING_ARRAY | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY;

        // adapterが対応してないとrequest_deviceで落ちるので、先にチェック
        if !adapter.features().contains(required_features) {
            return Err(PhotoEditorError::gpu_initialization_no_source());
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features,
            required_limits: adapter.limits().clone(),
            ..Default::default()
        }))
        .map_err(PhotoEditorError::gpu_initialization)?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // apply_adjustmentsのBindGroupと完全一致させる
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Main Bind Group Layout"),
            entries: &[
                // 0: image_in (texture_2d<f32>)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 1: image_out (storage_texture_2d<rgba32float, write>)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 2: base_params (uniform buffer)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: mask_texture (texture_2d_array<f32>)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // 4: params storage buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: tone LUT storage buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: brightness LUT storage buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7: hue LUT storage buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 8: saturation LUT storage buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 9: lightness LUT storage buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Main Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader_source = include_str!("wgpu_shader.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader Module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Main Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    pub fn device(&self) -> Arc<wgpu::Device> {
        self.device.clone()
    }
    pub fn queue(&self) -> Arc<wgpu::Queue> {
        self.queue.clone()
    }

    pub fn apply_adjustments(
        &self,
        original_image: &Image, // 入力テクスチャ(rgba32f想定)
        masks_data: &[&(GpuMask, EditParameters)], // マスクとパラメータのペア
    ) -> Result<Image, PhotoEditorError> {
        let t = time::Instant::now();
        let (width, height) = (original_image.width, original_image.height);
        let num_masks = masks_data.len() as u32;

        // 1. 各種バッファ用のベクタ準備
        let mut gpu_params = Vec::new();
        let mut all_tone_luts = Vec::new();
        let mut all_brightness_luts = Vec::new();
        let mut all_hue_luts = Vec::new();
        let mut all_sat_luts = Vec::new();
        let mut all_light_luts = Vec::new();

        for (_, params) in masks_data {
            // パラメータ変換
            gpu_params.push(GpuEditParameters {
                r_gain: 1.0 + 0.5 * (params.wb_temperature as f32 / 100.0),
                g_gain: 1.0 - 0.25 * (params.wb_tint as f32 / 100.0),
                b_gain: 1.0 - 0.5 * (params.wb_temperature as f32 / 100.0),
                vignette: params.vignette,
                lens_distortion: params.lens_distortion as f32,
            });

            // LUTの生成

            all_tone_luts.extend(self.create_tone_lut(params)?.to_vec());
            all_brightness_luts.extend(params.brightness_tone_curve.to_vec());
            all_hue_luts.extend(params.hue_tone_curve.to_vec());
            all_sat_luts.extend(params.saturation_tone_curve.to_vec());
            all_light_luts.extend(params.lightness_tone_curve.to_vec());
        }

        // 2. GPUバッファの作成
        let base_params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Base Params"),
                contents: bytemuck::bytes_of(&BaseParams {
                    width,
                    height,
                    num_masks,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let param_buf = self.create_storage_buffer("Params", &gpu_params);
        let tone_buf = self.create_storage_buffer("Tone LUT", &all_tone_luts);
        let bright_buf = self.create_storage_buffer("Bright LUT", &all_brightness_luts);
        let hue_buf = self.create_storage_buffer("Hue LUT", &all_hue_luts);
        let sat_buf = self.create_storage_buffer("Sat LUT", &all_sat_luts);
        let light_buf = self.create_storage_buffer("Light LUT", &all_light_luts);

        // 3. マスクテクスチャ配列の作成 (D2Array)
        let masks_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Masks Array"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: num_masks.max(1),
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // 4. 出力用テクスチャ
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let output_texture_view = output_tex.create_view(&Default::default());

        // 5. BindGroup生成
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Final Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&original_image.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&output_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: base_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&masks_tex.create_view(
                        &wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::D2Array),
                            ..Default::default()
                        },
                    )),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tone_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bright_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: hue_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: sat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: light_buf.as_entire_binding(),
                },
            ],
        });

        // 6. 実行
        let mut encoder = self.device.create_command_encoder(&Default::default());

        // 各レイヤーにマスクをコピー
        for (i, (mask, _)) in masks_data.iter().enumerate() {
            encoder.copy_texture_to_texture(
                mask.texture.as_image_copy(),
                wgpu::TexelCopyTextureInfo {
                    texture: &masks_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: i as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&self.pipeline); // WGSLの @compute fn main
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        let output_img =
            Image::from_texture(self.device(), self.queue(), output_tex, output_texture_view);

        println!("{:?}", time::Instant::now() - t);
        Ok(output_img)
    }

    // 補助関数: Storage Buffer作成用
    fn create_storage_buffer<T: bytemuck::Pod>(&self, label: &str, data: &[T]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    pub fn get_adapter_list() -> Vec<String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        instance
            .enumerate_adapters(wgpu::Backends::all())
            .into_iter()
            .map(|adapter| {
                format!(
                    "{}, {}",
                    adapter.get_info().name,
                    adapter.get_info().backend
                )
            })
            .collect()
    }

    fn create_tone_lut(&self, params: &EditParameters) -> Result<Array1<i32>, PhotoEditorError> {
        let x_lum = Array1::from_iter((0..65536).map(|i| i as f32 / 65535.0));
        let mut y_lum = x_lum.clone();

        y_lum = y_lum * (2.0f32).powf(params.exposure);

        let p5 = 0.05f32;
        let p25 = 0.25f32;
        let p50 = 0.50f32;
        let p75 = 0.75f32;
        let p95 = 0.95f32;
        let black_l = (p5 + (p50 - p5) * (params.black as f32 / 100.0)).clamp(0.0, 1.0);
        let shadow_l = (p25 + (p50 - p25) * (params.shadow as f32 / 100.0)).clamp(0.0, 1.0);
        let mid_l = p50;
        let highlight_l = (p75 - (p75 - p50) * (params.highlight as f32 / 100.0)).clamp(0.0, 1.0);
        let white_l = (p95 - (p95 - p50) * (params.white as f32 / 100.0)).clamp(0.0, 1.0);

        let mut points = vec![
            (0.0, 0.0),
            (black_l, (black_l + p5) / 2.0),
            (shadow_l, (shadow_l + p25) / 2.0),
            (mid_l, mid_l),
            (highlight_l, (highlight_l + p75) / 2.0),
            (white_l, (white_l + p95) / 2.0),
            (1.0, 1.0),
        ];
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        points.dedup_by_key(|p| p.0);

        let xs: Array1<f32> = points.iter().map(|p| p.0).collect();
        let ys: Array1<f32> = points.iter().map(|p| p.1).collect();

        let lum_mapped =
            interpolation::pchip_interpolate(&xs, &ys, &y_lum.mapv(|v| v.clamp(0.0, 1.0)))?;
        let c_factor = 1.0f32 + params.contrast as f32 / 100.0f32;
        let lum_contrasted = lum_mapped.mapv(|v| 0.5f32 + (v - 0.5f32) * c_factor);

        Ok(lum_contrasted.mapv(|v| (v.clamp(0.0f32, 1.0f32) * 65535.0f32) as i32))
    }
}
