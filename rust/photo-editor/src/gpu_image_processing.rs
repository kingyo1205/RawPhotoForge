use std::collections::HashMap;
use std::borrow::Cow;
use wgpu::util::DeviceExt;
use crate::errors::{GpuComputeError, GpuInitializationError, InterpolationError, PhotoEditorError};
use ndarray::{Array1, Array2};
use crate::EditParameters;
use bytemuck::{Pod, Zeroable};
use crate::interpolation;

//--------------------------------------------------------------------------------
// Uniform Structs (wgpu_shader.wgslと一致)
//--------------------------------------------------------------------------------

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

//--------------------------------------------------------------------------------
// GpuProcessor
//--------------------------------------------------------------------------------

pub struct GpuProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    width: u32,
    height: u32,
    base_params_buffer: wgpu::Buffer,
}

impl GpuProcessor {
    /// 新しいGpuProcessorインスタンスを作成します。
    /// WGPUデバイスを初期化し、シェーダーパイプラインをコンパイルします。
    pub fn new(width: u32, height: u32) -> Result<Self, PhotoEditorError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })).map_err(|_| PhotoEditorError::from(GpuInitializationError::Adapter))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits().clone(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            }
        )).map_err(GpuInitializationError::from)?;

        let shader_source = include_str!("wgpu_shader.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
        });

        let mut pipelines = HashMap::new();
        let pipeline_names = [
            "to_linear", "to_srgb", "clip_0_1", 
            "linear_srgb_to_oklch", "oklch_to_linear_srgb",
            "tone_curve_lut", "tone_curve_by_hue", "white_balance", "vignette_effect", "lens_distortion_effect"
        ];
        
        let wgsl_entry_points = [
            "to_linear", "to_srgb", "clip_0_1", 
            "rgb_to_oklch", "oklch_to_rgb", // シェーダー内の名前はそのまま
            "tone_curve_lut", "tone_curve_by_hue", "white_balance", "vignette_effect", "lens_distortion_effect"
        ];

        for (name, entry_point) in pipeline_names.iter().zip(wgsl_entry_points.iter()) {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &shader_module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            pipelines.insert(name.to_string(), pipeline);
        }

        let base_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Base Params Buffer"),
            contents: bytemuck::bytes_of(&BaseParams { width, height }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        Ok(Self { device, queue, pipelines, width, height, base_params_buffer })
    }

    /// 画像データに全ての編集を適用します。
    pub fn apply_adjustments(
        &self,
        image_data: &[f32],
        main_adjustments: &EditParameters,
        masks: &HashMap<String, (Array2<f32>, EditParameters)>,
    ) -> Result<Vec<f32>, PhotoEditorError> {
        let num_elements = (self.width * self.height * 3) as usize;

        let image_buffer_current = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Image Buffer Current"),
            contents: &bytemuck::cast_slice(image_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let image_buffer_scratch = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Image Buffer Scratch"),
            size: (num_elements * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Main Encoder") });

        // sRGB -> Linear
        self.to_linear(&mut encoder, &image_buffer_current);

        // Lens Distortion (if active, this is an out-of-place operation)
        if main_adjustments.lens_distortion != 0 {
            // Because lens distortion is not an in-place operation, we write to a scratch buffer.
            self.adjustment_lens_distortion(
                &mut encoder,
                &image_buffer_current,
                &image_buffer_scratch,
                main_adjustments,
            );
            // Then, we copy the result back to the current buffer so subsequent in-place operations work correctly.
            encoder.copy_buffer_to_buffer(&image_buffer_scratch, 0, &image_buffer_current, 0, image_buffer_scratch.size());
        }

        // Vignette (if active, and only for main adjustments)
        if main_adjustments.vignette != 0 {
            let mask_buffer = self.create_mask_buffer(None);
            self.adjustment_vignette(
                &mut encoder,
                &image_buffer_current,
                &mask_buffer,
                main_adjustments,
            );
        }

        // --- Linear RGB Adjustments (all in-place) ---
        let all_adjustments = std::iter::once((None, main_adjustments)).chain(
            masks.iter().map(|(_name, (mask, adj))| (Some(mask), adj))
        );

        for (mask_array, adjustments) in all_adjustments {
            let mask_buffer = self.create_mask_buffer(mask_array);
            self.adjustment_whitebalance(&mut encoder, &image_buffer_current, &mask_buffer, adjustments);
            self.adjustment_tone(&mut encoder, &image_buffer_current, &mask_buffer, adjustments)?;
            self.adjustment_tone_curve(&mut encoder, &image_buffer_current, &mask_buffer, &adjustments.brightness_tone_curve);
        }
        
        // Linear RGB -> Oklch (in-place)
        self.linear_srgb_to_oklch(&mut encoder, &image_buffer_current);

        // --- Oklch Adjustments (all in-place) ---
        let all_adjustments_oklch = std::iter::once((None, main_adjustments)).chain(
            masks.iter().map(|(_name, (mask, adj))| (Some(mask), adj))
        );
        for (mask_array, adjustments) in all_adjustments_oklch {
            let mask_buffer = self.create_mask_buffer(mask_array);
            self.adjustment_oklch_hue_tone_curve(&mut encoder, &image_buffer_current, &mask_buffer, &adjustments.hue_tone_curve);
            self.adjustment_oklch_saturation_tone_curve(&mut encoder, &image_buffer_current, &mask_buffer, &adjustments.saturation_tone_curve);
            self.adjustment_oklch_lightness_tone_curve(&mut encoder, &image_buffer_current, &mask_buffer, &adjustments.lightness_tone_curve);
        }

        // Oklch -> Linear RGB (in-place)
        self.oklch_to_linear_srgb(&mut encoder, &image_buffer_current);

        // Linear -> sRGB (in-place)
        self.to_srgb(&mut encoder, &image_buffer_current);
        
        // Final Clip (in-place)
        self.clip_0_1(&mut encoder, &image_buffer_current);

        // 結果をCPUに読み戻す
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (num_elements * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&image_buffer_current, 0, &output_buffer, 0, image_buffer_current.size());
        
        let submission_index = self.queue.submit(Some(encoder.finish()));

        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        self.device.poll(wgpu::PollType::Wait { submission_index: Some(submission_index), timeout: None }).map_err(|e| PhotoEditorError::GpuComputeError(e.into()))?;
        
        let mapped_result = rx.recv().map_err(|e| PhotoEditorError::GpuComputeError(GpuComputeError::ChannelReceive(e.to_string())))?;
        mapped_result.map_err(|e| PhotoEditorError::GpuComputeError(e.into()))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        
        Ok(result)
    }

    //--------------------------------------------------------------------------------
    // Private Helper Functions
    //--------------------------------------------------------------------------------

    /// シンプルなシェーダーパスを実行します (パラメータはwidth/heightのみ)。
    fn run_compute_shader(
        &self, 
        encoder: &mut wgpu::CommandEncoder, 
        pipeline_name: &str, 
        image_buffer: &wgpu::Buffer
    ) {
        let pipeline = self.pipelines.get(pipeline_name).unwrap();
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} bind group", pipeline_name)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: image_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.base_params_buffer.as_entire_binding() },
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(pipeline_name),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups_x = (self.width + 15) / 16;
        let workgroups_y = (self.height + 15) / 16;
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// マスク配列からGPUバッファを作成します。
    fn create_mask_buffer(&self, mask_array: Option<&Array2<f32>>) -> wgpu::Buffer {
        let num_pixels = (self.width * self.height) as usize;
        let mask_data: Vec<f32> = match mask_array {
            Some(arr) => arr.iter().cloned().collect(),
            None => vec![1.0f32; num_pixels], // マスクがない場合は全ピクセルに適用
        };
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask Buffer"),
            contents: &bytemuck::cast_slice(&mask_data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }
    
    //--------------------------------------------------------------------------------
    // Color Space Conversions & Clip
    //--------------------------------------------------------------------------------

    pub fn to_linear(&self, encoder: &mut wgpu::CommandEncoder, image_buffer: &wgpu::Buffer) {
        self.run_compute_shader(encoder, "to_linear", image_buffer);
    }
    
    pub fn to_srgb(&self, encoder: &mut wgpu::CommandEncoder, image_buffer: &wgpu::Buffer) {
        self.run_compute_shader(encoder, "to_srgb", image_buffer);
    }

    pub fn clip_0_1(&self, encoder: &mut wgpu::CommandEncoder, image_buffer: &wgpu::Buffer) {
        self.run_compute_shader(encoder, "clip_0_1", image_buffer);
    }
    
    pub fn linear_srgb_to_oklch(&self, encoder: &mut wgpu::CommandEncoder, image_buffer: &wgpu::Buffer) {
        self.run_compute_shader(encoder, "linear_srgb_to_oklch", image_buffer);
    }
    
    pub fn oklch_to_linear_srgb(&self, encoder: &mut wgpu::CommandEncoder, image_buffer: &wgpu::Buffer) {
        self.run_compute_shader(encoder, "oklch_to_linear_srgb", image_buffer);
    }

    //--------------------------------------------------------------------------------
    // Adjustments
    //--------------------------------------------------------------------------------
    
    /// ホワイトバランス調整を適用します。
    pub fn adjustment_whitebalance(&self, encoder: &mut wgpu::CommandEncoder, image_buffer: &wgpu::Buffer, mask_buffer: &wgpu::Buffer, adjustments: &EditParameters) {
        if adjustments.wb_temperature != 0 || adjustments.wb_tint != 0 {
             let r_gain = 1.0 + 0.5 * (adjustments.wb_temperature as f32 / 100.0);
             let b_gain = 1.0 - 0.5 * (adjustments.wb_temperature as f32 / 100.0);
             let g_gain = 1.0 - 0.25 * (adjustments.wb_tint as f32 / 100.0);
             let params = WhiteBalanceParams { width: self.width, height: self.height, r_gain, g_gain, b_gain };
             
             let pipeline = self.pipelines.get("white_balance").unwrap();
             let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WhiteBalance Params Buffer"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
             });
             let bind_group_layout = pipeline.get_bind_group_layout(0);
             let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                 label: Some("white_balance bind group"), layout: &bind_group_layout,
                 entries: &[
                     wgpu::BindGroupEntry { binding: 0, resource: image_buffer.as_entire_binding() },
                     wgpu::BindGroupEntry { binding: 3, resource: mask_buffer.as_entire_binding() },
                     wgpu::BindGroupEntry { binding: 7, resource: params_buffer.as_entire_binding() },
                 ],
             });
             let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
             compute_pass.set_pipeline(pipeline);
             compute_pass.set_bind_group(0, &bind_group, &[]);
             compute_pass.dispatch_workgroups((self.width + 15) / 16, (self.height + 15) / 16, 1);
        }
    }

    /// トーン調整（露出、コントラストなど）を適用します。
    pub fn adjustment_tone(&self, encoder: &mut wgpu::CommandEncoder, image_buffer: &wgpu::Buffer, mask_buffer: &wgpu::Buffer, adjustments: &EditParameters) -> Result<(), PhotoEditorError> {
        let lut = self.create_tone_lut(adjustments)?;
        self.adjustment_tone_curve(encoder, image_buffer, mask_buffer, &lut);
        Ok(())
    }

    /// 明るさトーンカーブを適用します。
    pub fn adjustment_tone_curve(&self, encoder: &mut wgpu::CommandEncoder, image_buffer: &wgpu::Buffer, mask_buffer: &wgpu::Buffer, curve_data: &Array1<i32>) {
        let pipeline = self.pipelines.get("tone_curve_lut").unwrap();
        let curve_data_f32: Vec<f32> = curve_data.iter().map(|&x| x as f32).collect();
        let curve_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Curve Buffer"), contents: &bytemuck::cast_slice(&curve_data_f32), usage: wgpu::BufferUsages::STORAGE,
        });
        let channels = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Channels Buffer"), contents: bytemuck::cast_slice(&[0u32, 1, 2]), usage: wgpu::BufferUsages::STORAGE,
        });
        let params = ToneCurveParams { width: self.width, height: self.height, channel_count: 3 };
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ToneCurveParams Buffer"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone_curve_lut bind group"), layout: &bind_group_layout,
            entries: &[
                 wgpu::BindGroupEntry { binding: 0, resource: image_buffer.as_entire_binding() },
                 wgpu::BindGroupEntry { binding: 2, resource: curve_buffer.as_entire_binding() },
                 wgpu::BindGroupEntry { binding: 3, resource: mask_buffer.as_entire_binding() },
                 wgpu::BindGroupEntry { binding: 4, resource: channels.as_entire_binding() },
                 wgpu::BindGroupEntry { binding: 5, resource: params_buffer.as_entire_binding() },
            ],
        });
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((self.width + 15) / 16, (self.height + 15) / 16, 1);
    }

    /// Oklchの色相トーンカーブを適用します。
    pub fn adjustment_oklch_hue_tone_curve(&self, encoder: &mut wgpu::CommandEncoder, image_buffer: &wgpu::Buffer, mask_buffer: &wgpu::Buffer, curve_data: &Array1<i32>) {
        let pipeline = self.pipelines.get("tone_curve_lut").unwrap();
        let curve_data_f32: Vec<f32> = curve_data.iter().map(|&x| x as f32).collect();
        let curve_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hue Curve Buffer"), contents: &bytemuck::cast_slice(&curve_data_f32), usage: wgpu::BufferUsages::STORAGE,
        });
        let channels = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hue Channel Buffer"), contents: bytemuck::cast_slice(&[2u32]), usage: wgpu::BufferUsages::STORAGE,
        });
        let params = ToneCurveParams { width: self.width, height: self.height, channel_count: 1 };
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hue ToneCurveParams Buffer"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hue_tone_curve bind group"), layout: &bind_group_layout,
            entries: &[
                 wgpu::BindGroupEntry { binding: 0, resource: image_buffer.as_entire_binding() },
                 wgpu::BindGroupEntry { binding: 2, resource: curve_buffer.as_entire_binding() },
                 wgpu::BindGroupEntry { binding: 3, resource: mask_buffer.as_entire_binding() },
                 wgpu::BindGroupEntry { binding: 4, resource: channels.as_entire_binding() },
                 wgpu::BindGroupEntry { binding: 5, resource: params_buffer.as_entire_binding() },
            ],
        });
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((self.width + 15) / 16, (self.height + 15) / 16, 1);
    }

    /// Oklchの色相別彩度トーンカーブを適用します。
    pub fn adjustment_oklch_saturation_tone_curve(&self, encoder: &mut wgpu::CommandEncoder, image_buffer: &wgpu::Buffer, mask_buffer: &wgpu::Buffer, curve_data: &Array1<i32>) {
        self.apply_curve_by_hue(encoder, image_buffer, curve_data, mask_buffer, 2, 1) // ch_hue: 2 (Lchのh), ch_target: 1 (Lchのc)
    }

    /// Oklchの色相別輝度トーンカーブを適用します。
    pub fn adjustment_oklch_lightness_tone_curve(&self, encoder: &mut wgpu::CommandEncoder, image_buffer: &wgpu::Buffer, mask_buffer: &wgpu::Buffer, curve_data: &Array1<i32>) {
        self.apply_curve_by_hue(encoder, image_buffer, curve_data, mask_buffer, 2, 0) // ch_hue: 2 (Lchのh), ch_target: 0 (LchのL)
    }
    
    /// 周辺光量落ち（ヴィネット）効果を適用します。
    pub fn adjustment_vignette(&self, encoder: &mut wgpu::CommandEncoder, image_buffer: &wgpu::Buffer, mask_buffer: &wgpu::Buffer, adjustments: &EditParameters) {
        if adjustments.vignette != 0 {
            let strength = (-adjustments.vignette as f32 / 100.0) * 2.0;
            let params = VignetteParams { width: self.width, height: self.height, strength };
            let pipeline = self.pipelines.get("vignette_effect").unwrap();
            let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
               label: Some("Vignette Params Buffer"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vignette bind group"), layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: image_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: mask_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: params_buffer.as_entire_binding() },
                ],
            });
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((self.width + 15) / 16, (self.height + 15) / 16, 1);
        }
    }

    /// レンズ歪み補正を適用します。
    pub fn adjustment_lens_distortion(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        image_buffer_in: &wgpu::Buffer,
        image_buffer_out: &wgpu::Buffer,
        adjustments: &EditParameters,
    ) {
        if adjustments.lens_distortion != 0 {
            // スライダーの値 -100..100 を歪み係数に変換
            // k の範囲を調整して、より自然な見た目にする
            let strength_scaled = adjustments.lens_distortion as f32 / 100.0 * -0.5; // 例: -0.5 から 0.5 (負の値で樽型補正、正の値で糸巻き型補正)

            let params = LensDistortionParams {
                width: self.width,
                height: self.height,
                strength: strength_scaled,
            };

            let pipeline = self.pipelines.get("lens_distortion_effect").unwrap();
            let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LensDistortion Params Buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lens_distortion bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 9, resource: params_buffer.as_entire_binding() }, // binding 9 for params
                    wgpu::BindGroupEntry { binding: 11, resource: image_buffer_in.as_entire_binding() }, // image_in
                    wgpu::BindGroupEntry { binding: 12, resource: image_buffer_out.as_entire_binding() }, // image_out
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((self.width + 15) / 16, (self.height + 15) / 16, 1);
        }
    }

    /// Oklchの色相を基準にカーブを適用するヘルパー関数です。
    fn apply_curve_by_hue(&self, encoder: &mut wgpu::CommandEncoder, image_buffer: &wgpu::Buffer, curve_data: &Array1<i32>, mask_buffer: &wgpu::Buffer, ch_hue: u32, ch_target: u32) {
        let pipeline = self.pipelines.get("tone_curve_by_hue").unwrap();
        let curve_data_f32: Vec<f32> = curve_data.iter().map(|&x| x as f32).collect();
        let curve_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CurveByHue Buffer"), contents: &bytemuck::cast_slice(&curve_data_f32), usage: wgpu::BufferUsages::STORAGE,
        });
        let params = ToneCurveByHueParams { width: self.width, height: self.height, ch_hue, ch_target };
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ToneCurveByHueParams Buffer"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone_curve_by_hue bind group"), layout: &bind_group_layout,
            entries: &[
                 wgpu::BindGroupEntry { binding: 0, resource: image_buffer.as_entire_binding() },
                 wgpu::BindGroupEntry { binding: 2, resource: curve_buffer.as_entire_binding() },
                 wgpu::BindGroupEntry { binding: 3, resource: mask_buffer.as_entire_binding() },
                 wgpu::BindGroupEntry { binding: 6, resource: params_buffer.as_entire_binding() },
            ],
        });
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((self.width + 15) / 16, (self.height + 15) / 16, 1);
    }

    /// トーン調整用のLUT（ルックアップテーブル）を作成します。
    fn create_tone_lut(&self, params: &EditParameters) -> Result<Array1<i32>, InterpolationError> {
        let x_lum = Array1::from_iter((0..65536).map(|i| i as f32 / 65535.0));
        let mut y_lum = x_lum.clone();

        // EV
        y_lum = y_lum * (2.0f32).powf(params.exposure);

        // Tone Curve
        let p5  = 0.05f32;
        let p25 = 0.25f32;
        let p50 = 0.50f32;
        let p75 = 0.75f32;
        let p95 = 0.95f32;

        let black_l     = p5  + (p50 - p5)  * (params.black as f32 / 100.0);
        let shadow_l    = p25 + (p50 - p25) * (params.shadow as f32 / 100.0);
        let mid_l       = p50;
        let highlight_l = p75 - (p75 - p50) * (params.highlight as f32 / 100.0);
        let white_l     = p95 - (p95 - p50) * (params.white as f32 / 100.0);

        let black_l_clamped = black_l.clamp(0.0, 1.0);
        let shadow_l_clamped = shadow_l.clamp(0.0, 1.0);
        let mid_l_clamped = mid_l.clamp(0.0, 1.0);
        let highlight_l_clamped = highlight_l.clamp(0.0, 1.0);
        let white_l_clamped = white_l.clamp(0.0, 1.0);

        let mut points = vec![
            (0.0, 0.0),
            (black_l_clamped, (black_l_clamped + p5) / 2.0),
            (shadow_l_clamped, (shadow_l_clamped + p25) / 2.0),
            (mid_l_clamped, mid_l_clamped),
            (highlight_l_clamped, (highlight_l_clamped + p75) / 2.0),
            (white_l_clamped, (white_l_clamped + p95) / 2.0),
            (1.0, 1.0),
        ];

        // x値でソート
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // x値が重複している点を処理し、各x値に対して最後のy値のみを保持する
        let mut processed_points: Vec<(f32, f32)> = Vec::new();
        for point in points {
            if let Some(last) = processed_points.last_mut() {
                if last.0 == point.0 {
                    // x値が同じ場合は、y値を上書きする
                    last.1 = point.1;
                } else {
                    // x値が異なる場合は新しい点を追加
                    processed_points.push(point);
                }
            } else {
                // 最初の点を追加
                processed_points.push(point);
            }
        }

        let xs: Array1<f32> = processed_points.iter().map(|p| p.0).collect();
        let ys: Array1<f32> = processed_points.iter().map(|p| p.1).collect();

        let lum_mapped = interpolation::pchip_interpolate(&xs, &ys, &y_lum.mapv(|v| v.clamp(0.0, 1.0)))?;

        // Contrast
        let c_factor = 1.0f32 + params.contrast as f32 / 100.0f32;
        let lum_contrasted = lum_mapped.mapv(|v| 0.5f32 + (v - 0.5f32) * c_factor);
        
        Ok(lum_contrasted.mapv(|v| (v.clamp(0.0f32, 1.0f32) * 65535.0f32) as i32))
    }
}