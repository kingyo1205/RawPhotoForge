use std::{collections::HashMap, time};
use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use crate::errors::PhotoEditorError;
use ndarray::{Array1};
use crate::{EditParameters, Image, GpuMask};
use bytemuck::{Pod, Zeroable};
use crate::interpolation;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BaseParams {
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ToneCurveParams {
    width: u32,
    height: u32,
    channel_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ToneCurveByHueParams {
    width: u32,
    height: u32,
    ch_hue: u32,
    ch_target: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WhiteBalanceParams {
    width: u32,
    height: u32,
    r_gain: f32,
    g_gain: f32,
    b_gain: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VignetteParams {
    width: u32,
    height: u32,
    strength: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LensDistortionParams {
    width: u32,
    height: u32,
    strength: f32,
}

pub struct GpuProcessor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    empty_mask_view: wgpu::TextureView,
}

impl GpuProcessor {
    pub fn new(adapter_index: usize) -> Result<Self, PhotoEditorError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .into_iter()
            .nth(adapter_index)
            .ok_or_else(|| PhotoEditorError::gpu_initialization_no_source())?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::TEXTURE_BINDING_ARRAY | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                required_limits: adapter.limits().clone(),
                ..Default::default()
            },
        )).map_err(PhotoEditorError::gpu_initialization)?;
        
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Main Bind Group Layout"),
            entries: &[
                // image_in (texture_2d)
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
                // image_out (storage_texture_2d)
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
                // base_params
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // curve
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // mask_texture
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // channels
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // other params...
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 9, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 10, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Main Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader_source = include_str!("wgpu_shader.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
        });

        let mut pipelines = HashMap::new();
        let pipeline_names = [
            "to_linear", "to_srgb", "clip_0_1", 
            "rgb_to_oklch", "oklch_to_rgb",
            "tone_curve_lut", "tone_curve_by_hue", "white_balance", "vignette_effect", "lens_distortion_effect"
        ];

        for name in pipeline_names {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(name),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None
            });
            pipelines.insert(name.to_string(), pipeline);
        }

        let empty_mask_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Empty Mask Texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            empty_mask_texture.as_image_copy(),
            bytemuck::cast_slice(&[0.0f32]),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: None },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 }
        );
        let empty_mask_view = empty_mask_texture.create_view(&Default::default());


        Ok(Self { device, queue, pipelines, bind_group_layout, empty_mask_view })
    }

    pub fn device(&self) -> Arc<wgpu::Device> { self.device.clone() }
    pub fn queue(&self) -> Arc<wgpu::Queue> { self.queue.clone() }

    pub fn apply_adjustments(
        &self,
        original_image: &Image,
        masks: &HashMap<String, (GpuMask, EditParameters)>,
    ) -> Result<Image, PhotoEditorError> {
        let t = time::Instant::now();
        let (width, height) = (original_image.width, original_image.height);

        let mut src_tex = original_image.clone();
        let mut dst_tex = self.create_scratch_texture(width, height);

        let base_params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Base Params Buffer"),
            contents: bytemuck::bytes_of(&BaseParams { width, height }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Main Encoder") });

        // sRGB -> Linear
        self.run_pass(&mut encoder, "to_linear", &src_tex, &mut dst_tex, &base_params_buffer, None, Some(&self.empty_mask_view));
        std::mem::swap(&mut src_tex, &mut dst_tex);

        let (main_mask, main_adjustments) = masks.get("main")
            .ok_or_else(|| PhotoEditorError::MaskNotFound("main mask not found".to_string()))?;

        // Lens Distortion (main only)
        if main_adjustments.lens_distortion != 0 {
            self.adjustment_lens_distortion(&mut encoder, &src_tex, &mut dst_tex, main_adjustments);
            std::mem::swap(&mut src_tex, &mut dst_tex);
        }

        // Vignette (main only)
        if main_adjustments.vignette != 0 {
             self.adjustment_vignette(&mut encoder, &src_tex, &mut dst_tex, main_adjustments, &main_mask.view);
             std::mem::swap(&mut src_tex, &mut dst_tex);
        }

        // --- Linear RGB Adjustments ---
        for (_, (mask, adjustments)) in masks.iter() {
            self.adjustment_whitebalance(&mut encoder, &src_tex, &mut dst_tex, adjustments, &mask.view);
            std::mem::swap(&mut src_tex, &mut dst_tex);
            
            let lut = self.create_tone_lut(adjustments)?;
            self.adjustment_tone_curve(&mut encoder, &src_tex, &mut dst_tex, &lut, &mask.view, &[0, 1, 2]);
            std::mem::swap(&mut src_tex, &mut dst_tex);

            self.adjustment_tone_curve(&mut encoder, &src_tex, &mut dst_tex, &adjustments.brightness_tone_curve, &mask.view, &[0, 1, 2]);
            std::mem::swap(&mut src_tex, &mut dst_tex);
        }

        // Linear RGB -> Oklch
        self.run_pass(&mut encoder, "rgb_to_oklch", &src_tex, &mut dst_tex, &base_params_buffer, None, Some(&self.empty_mask_view));
        std::mem::swap(&mut src_tex, &mut dst_tex);

        // --- Oklch Adjustments ---
        for (_, (mask, adjustments)) in masks.iter() {
             self.adjustment_oklch_hue_tone_curve(&mut encoder, &src_tex, &mut dst_tex, &adjustments.hue_tone_curve, &mask.view);
             std::mem::swap(&mut src_tex, &mut dst_tex);
             self.adjustment_oklch_saturation_tone_curve(&mut encoder, &src_tex, &mut dst_tex, &adjustments.saturation_tone_curve, &mask.view);
             std::mem::swap(&mut src_tex, &mut dst_tex);
             self.adjustment_oklch_lightness_tone_curve(&mut encoder, &src_tex, &mut dst_tex, &adjustments.lightness_tone_curve, &mask.view);
             std::mem::swap(&mut src_tex, &mut dst_tex);
        }
        
        // Oklch -> Linear RGB
        self.run_pass(&mut encoder, "oklch_to_rgb", &src_tex, &mut dst_tex, &base_params_buffer, None, Some(&self.empty_mask_view));
        std::mem::swap(&mut src_tex, &mut dst_tex);

        // Linear -> sRGB
        self.run_pass(&mut encoder, "to_srgb", &src_tex, &mut dst_tex, &base_params_buffer, None, Some(&self.empty_mask_view));
        std::mem::swap(&mut src_tex, &mut dst_tex);
        
        // Final Clip
        self.run_pass(&mut encoder, "clip_0_1", &src_tex, &mut dst_tex, &base_params_buffer, None, Some(&self.empty_mask_view));
        std::mem::swap(&mut src_tex, &mut dst_tex);

        self.queue.submit(Some(encoder.finish()));
        println!("{:?}", time::Instant::now() - t);
        // The final result is in src_tex
        Ok(src_tex)
    }

    fn create_scratch_texture(&self, width: u32, height: u32) -> Image {
        Image::new(self.device(), self.queue(), &[], width, height) // Data is empty as it will be written to
    }
    
    fn run_pass(&self, encoder: &mut wgpu::CommandEncoder, pipeline_name: &str, src: &Image, dst: &Image, base_params_buffer: &wgpu::Buffer, extra_uniforms: Option<&wgpu::Buffer>, mask_view: Option<&wgpu::TextureView>) {
        let pipeline = self.pipelines.get(pipeline_name).unwrap();
        let src_view = src.texture.create_view(&Default::default());
        let dst_view = dst.texture.create_view(&Default::default());

        let empty_buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE, mapped_at_creation: false });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", pipeline_name)),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 2, resource: base_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: empty_buffer.as_entire_binding() }, // curve
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(mask_view.unwrap_or(&self.empty_mask_view)) },
                wgpu::BindGroupEntry { binding: 5, resource: empty_buffer.as_entire_binding() }, // channels
                wgpu::BindGroupEntry { binding: 6, resource: extra_uniforms.unwrap_or(&empty_buffer).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: extra_uniforms.unwrap_or(&empty_buffer).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: extra_uniforms.unwrap_or(&empty_buffer).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: extra_uniforms.unwrap_or(&empty_buffer).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: extra_uniforms.unwrap_or(&empty_buffer).as_entire_binding() },
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((src.width + 15) / 16, (src.height + 15) / 16, 1);
    }
    
    // --- Specific Adjustment Implementations ---
    
    fn adjustment_whitebalance(&self, encoder: &mut wgpu::CommandEncoder, src: &Image, dst: &Image, adjustments: &EditParameters, mask_view: &wgpu::TextureView) {
        
        let r_gain = 1.0 + 0.5 * (adjustments.wb_temperature as f32 / 100.0);
        let b_gain = 1.0 - 0.5 * (adjustments.wb_temperature as f32 / 100.0);
        let g_gain = 1.0 - 0.25 * (adjustments.wb_tint as f32 / 100.0);
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("WhiteBalance Params"),
            contents: bytemuck::bytes_of(&WhiteBalanceParams { width: src.width, height: src.height, r_gain, g_gain, b_gain }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        self.run_specific_pass(encoder, "white_balance", src, dst, mask_view, &params_buffer, None, None);
    }

    fn adjustment_tone_curve(&self, encoder: &mut wgpu::CommandEncoder, src: &Image, dst: &Image, curve_data: &Array1<i32>, mask_view: &wgpu::TextureView, channels: &[u32]) {
        let curve_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Curve Buffer"), contents: &bytemuck::cast_slice(&curve_data.mapv(|x| x as f32).to_vec()), usage: wgpu::BufferUsages::STORAGE,
        });
        let channels_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Channels Buffer"), contents: bytemuck::cast_slice(channels), usage: wgpu::BufferUsages::STORAGE,
        });
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ToneCurve Params"),
            contents: bytemuck::bytes_of(&ToneCurveParams { width: src.width, height: src.height, channel_count: channels.len() as u32 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        self.run_specific_pass(encoder, "tone_curve_lut", src, dst, mask_view, &params_buffer, Some(&curve_buffer), Some(&channels_buffer));
    }

    
    fn adjustment_oklch_hue_tone_curve(&self, encoder: &mut wgpu::CommandEncoder, src: &Image, dst: &Image, curve_data: &Array1<i32>, mask_view: &wgpu::TextureView) {
        self.adjustment_tone_curve(encoder, src, dst, curve_data, mask_view, &[2]);
    }
    
    fn adjustment_oklch_saturation_tone_curve(&self, encoder: &mut wgpu::CommandEncoder, src: &Image, dst: &Image, curve_data: &Array1<i32>, mask_view: &wgpu::TextureView) {
        self.apply_curve_by_hue(encoder, src, dst, curve_data, mask_view, 2, 1);
    }
    
    fn adjustment_oklch_lightness_tone_curve(&self, encoder: &mut wgpu::CommandEncoder, src: &Image, dst: &Image, curve_data: &Array1<i32>, mask_view: &wgpu::TextureView) {
        self.apply_curve_by_hue(encoder, src, dst, curve_data, mask_view, 2, 0);
    }

    fn apply_curve_by_hue(&self, encoder: &mut wgpu::CommandEncoder, src: &Image, dst: &Image, curve_data: &Array1<i32>, mask_view: &wgpu::TextureView, ch_hue: u32, ch_target: u32) {
        let curve_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CurveByHue Buffer"), contents: &bytemuck::cast_slice(&curve_data.mapv(|x| x as f32).to_vec()), usage: wgpu::BufferUsages::STORAGE,
        });
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ToneCurveByHue Params"),
            contents: bytemuck::bytes_of(&ToneCurveByHueParams { width: src.width, height: src.height, ch_hue, ch_target }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        self.run_specific_pass(encoder, "tone_curve_by_hue", src, dst, mask_view, &params_buffer, Some(&curve_buffer), None);
    }

    fn adjustment_vignette(&self, encoder: &mut wgpu::CommandEncoder, src: &Image, dst: &Image, adjustments: &EditParameters, mask_view: &wgpu::TextureView) {
        let strength = (-adjustments.vignette as f32 / 100.0) * 2.0;
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vignette Params"),
            contents: bytemuck::bytes_of(&VignetteParams { width: src.width, height: src.height, strength }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        self.run_specific_pass(encoder, "vignette_effect", src, dst, mask_view, &params_buffer, None, None);
    }

    fn adjustment_lens_distortion(&self, encoder: &mut wgpu::CommandEncoder, src: &Image, dst: &Image, adjustments: &EditParameters) {
        if adjustments.lens_distortion == 0 {
            self.copy_texture(encoder, &src.texture, &dst.texture);
            return;
        }
        let strength_scaled = adjustments.lens_distortion as f32 / 100.0 * -0.5;
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LensDistortion Params"),
            contents: bytemuck::bytes_of(&LensDistortionParams { width: src.width, height: src.height, strength: strength_scaled }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        self.run_specific_pass(encoder, "lens_distortion_effect", src, dst, &self.empty_mask_view, &params_buffer, None, None);
    }

    fn run_specific_pass(&self, encoder: &mut wgpu::CommandEncoder, pipeline_name: &str, src: &Image, dst: &Image, mask_view: &wgpu::TextureView, params: &wgpu::Buffer, curve: Option<&wgpu::Buffer>, channels: Option<&wgpu::Buffer>) {
        let pipeline = self.pipelines.get(pipeline_name).unwrap();
        let src_view = src.texture.create_view(&Default::default());
        let dst_view = dst.texture.create_view(&Default::default());
        let empty_buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", pipeline_name)),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: curve.unwrap_or(&empty_buffer).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(mask_view) },
                wgpu::BindGroupEntry { binding: 5, resource: channels.unwrap_or(&empty_buffer).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: params.as_entire_binding() },
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((src.width + 15) / 16, (src.height + 15) / 16, 1);
    }
    
    fn copy_texture(&self, encoder: &mut wgpu::CommandEncoder, src: &wgpu::Texture, dst: &wgpu::Texture) {
        encoder.copy_texture_to_texture(src.as_image_copy(), dst.as_image_copy(), src.size());
    }

    pub fn get_adapter_list() -> Vec<String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        instance.enumerate_adapters(wgpu::Backends::all()).into_iter().map(|adapter| format!("{}, {}", adapter.get_info().name, adapter.get_info().backend)).collect()
    }

    fn create_tone_lut(&self, params: &EditParameters) -> Result<Array1<i32>, PhotoEditorError> {
        let x_lum = Array1::from_iter((0..65536).map(|i| i as f32 / 65535.0));
        let mut y_lum = x_lum.clone();

        y_lum = y_lum * (2.0f32).powf(params.exposure);

        let p5 = 0.05f32; let p25 = 0.25f32; let p50 = 0.50f32; let p75 = 0.75f32; let p95 = 0.95f32;
        let black_l = (p5 + (p50 - p5) * (params.black as f32 / 100.0)).clamp(0.0, 1.0);
        let shadow_l = (p25 + (p50 - p25) * (params.shadow as f32 / 100.0)).clamp(0.0, 1.0);
        let mid_l = p50;
        let highlight_l = (p75 - (p75 - p50) * (params.highlight as f32 / 100.0)).clamp(0.0, 1.0);
        let white_l = (p95 - (p95 - p50) * (params.white as f32 / 100.0)).clamp(0.0, 1.0);

        let mut points = vec![(0.0, 0.0), (black_l, (black_l + p5) / 2.0), (shadow_l, (shadow_l + p25) / 2.0), (mid_l, mid_l), (highlight_l, (highlight_l + p75) / 2.0), (white_l, (white_l + p95) / 2.0), (1.0, 1.0)];
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        points.dedup_by_key(|p| p.0);

        let xs: Array1<f32> = points.iter().map(|p| p.0).collect();
        let ys: Array1<f32> = points.iter().map(|p| p.1).collect();

        let lum_mapped = interpolation::pchip_interpolate(&xs, &ys, &y_lum.mapv(|v| v.clamp(0.0, 1.0)))?;
        let c_factor = 1.0f32 + params.contrast as f32 / 100.0f32;
        let lum_contrasted = lum_mapped.mapv(|v| 0.5f32 + (v - 0.5f32) * c_factor);
        
        Ok(lum_contrasted.mapv(|v| (v.clamp(0.0f32, 1.0f32) * 65535.0f32) as i32))
    }
}