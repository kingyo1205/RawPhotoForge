// lib.rs


pub mod metadata;
pub mod errors;
pub mod image;
pub mod gpu_image_processing;
pub mod interpolation;

use std::collections::HashMap;
use std::sync::Arc;
use ndarray::{Array1, Array2};
use errors::{PhotoEditorError, InterpolationError};
use image::Image;
use metadata::Exif;
pub use gpu_image_processing::GpuProcessor;
pub use crate::image::ImageFormat;

const CURVE_RESOLUTION: usize = 65536;

#[derive(Debug, Clone)]
pub struct EditParameters {
    // Tone
    pub exposure: f32,
    pub contrast: i32,
    pub shadow: i32,
    pub highlight: i32,
    pub black: i32,
    pub white: i32,
    // White Balance
    pub wb_temperature: i32,
    pub wb_tint: i32,
    // Vignette
    pub vignette: i32,
    // Lens Distortion
    pub lens_distortion: i32,
    // Mask Range
    pub mask_range: f32,
    // Curves
    pub brightness_tone_curve: Array1<i32>,
    pub hue_tone_curve: Array1<i32>,
    pub saturation_tone_curve: Array1<i32>,
    pub lightness_tone_curve: Array1<i32>,
}

impl Default for EditParameters {
    fn default() -> Self {
        Self {
            exposure: 0.0,
            contrast: 0,
            shadow: 0,
            highlight: 0,
            black: 0,
            white: 0,
            wb_temperature: 0,
            wb_tint: 0,
            vignette: 0,
            mask_range: 0.0, // デフォルト値を0.0に変更
            lens_distortion: 0,
            brightness_tone_curve: Array1::from_iter((0..CURVE_RESOLUTION).map(|i| i as i32)),
            hue_tone_curve: Array1::from_iter((0..CURVE_RESOLUTION).map(|i| i as i32)),
            saturation_tone_curve: Array1::from_elem(CURVE_RESOLUTION, 65535i32),
            lightness_tone_curve: Array1::from_elem(CURVE_RESOLUTION, 65535i32),
        }
    }
}

pub struct GpuMask {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}


pub struct PhotoEditor {
    pub image: Image,
    pub original_image: Image,
    pub exif: Exif,
    pub image_format: image::ImageFormat,
    pub masks: HashMap<String, (GpuMask, EditParameters)>,
    gpu_processor: Arc<GpuProcessor>,
}

impl PhotoEditor {
    pub fn new(gpu_processor: Arc<GpuProcessor>, file_data: &[u8], image_format: image::ImageFormat) -> Result<PhotoEditor, PhotoEditorError> {
        let (image, exif) = image::read_image(
            gpu_processor.device(),
            gpu_processor.queue(),
            file_data,
            &image_format,
        )?;
        let original_image = image.clone();

        let mut masks = HashMap::new();
        let main_mask = Self::create_gpu_mask(
            gpu_processor.device(),
            gpu_processor.queue(),
            image.width,
            image.height,
            None // None for a full mask of 1.0s
        );
        masks.insert("main".to_string(), (main_mask, EditParameters::default()));

        Ok(PhotoEditor {
            image,
            original_image,
            exif,
            image_format,
            masks,
            gpu_processor,
        })
    }
    
    fn create_gpu_mask(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, width: u32, height: u32, data: Option<&[f32]>) -> GpuMask {
        let texture_size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Mask Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let mask_data = match data {
            Some(d) => d.to_vec(),
            None => vec![1.0f32; (width * height) as usize],
        };

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&mask_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            texture_size,
        );
        
        let view = texture.create_view(&Default::default());
        
        GpuMask { texture, view }
    }


    pub fn get_exif_hashmap(&self) -> HashMap<String, String> {
        self.exif.to_hashmap()
    }

    pub fn save(&self, image_format: &ImageFormat) -> Result<Vec<u8>, PhotoEditorError> {
        Ok(image::write_image(&self.image, image_format)?)
    }

    pub fn reset(&mut self) {
        self.image = self.original_image.clone();
        self.masks.clear();
        let main_mask = Self::create_gpu_mask(
            self.gpu_processor.device(),
            self.gpu_processor.queue(),
            self.image.width,
            self.image.height,
            None
        );
        self.masks.insert("main".to_string(), (main_mask, EditParameters::default()));
    }
    
    fn get_adjustment_set(&mut self, mask_name: Option<&str>) -> Result<&mut EditParameters, PhotoEditorError> {
        let name = mask_name.unwrap_or("main");
        self.masks.get_mut(name)
            .map(|(_, params)| params)
            .ok_or_else(|| PhotoEditorError::MaskNotFound(format!("The specified mask '{}' does not exist.", name)))
    }
    
    // --- Setter methods ---
    pub fn set_whitebalance(&mut self, temperature: i32, tint: i32, mask_name: Option<&str>) -> Result<(), PhotoEditorError> {
        let adjustments = self.get_adjustment_set(mask_name)?;
        adjustments.wb_temperature = temperature.clamp(-100, 100);
        adjustments.wb_tint = tint.clamp(-100, 100);
        Ok(())
    }
    
    pub fn set_vignette(&mut self, value: i32) -> Result<(), PhotoEditorError> {
        self.get_adjustment_set(None)?.vignette = value.clamp(-100, 100);
        Ok(())
    }

    pub fn set_lens_distortion_correction(&mut self, value: i32) -> Result<(), PhotoEditorError> {
        self.get_adjustment_set(None)?.lens_distortion = value.clamp(-100, 100);
        Ok(())
    }

    /// 写真の明るさ関連の調整を一括で設定します。
    pub fn set_tone(
        &mut self,
        exposure: f32,
        contrast: i32,
        shadow: i32,
        highlight: i32,
        black: i32,
        white: i32,
        mask_name: Option<&str>,
    ) -> Result<(), PhotoEditorError> {
        let adjustments = self.get_adjustment_set(mask_name)?;

        adjustments.exposure = exposure.clamp(-10.0, 10.0);
        adjustments.contrast = contrast.clamp(-100, 100);
        adjustments.shadow = shadow.clamp(-100, 100);
        adjustments.highlight = highlight.clamp(-100, 100);
        adjustments.black = black.clamp(-100, 100);
        adjustments.white = white.clamp(-100, 100);

        Ok(())
    }

    pub fn set_brightness_tone_curve(
        &mut self,
        curve: Option<Array1<i32>>,
        control_points_x: Option<Array1<i32>>,
        control_points_y: Option<Array1<i32>>,
        mask_name: Option<&str>,
    ) -> Result<(), PhotoEditorError> {
        if curve.is_none() && control_points_x.is_none() {
             return Err(InterpolationError::MissingCurveOrControlPoints.into());
        }

        let final_curve: Array1<i32>;

        if let Some(c) = curve {
            if c.len() != CURVE_RESOLUTION {
                return Err(InterpolationError::InvalidCurveLength { expected: CURVE_RESOLUTION, actual: c.len() }.into());
            }
            final_curve = c;
        } else {
            let x = control_points_x.ok_or(InterpolationError::MissingControlPoints)?;
            let y = control_points_y.ok_or(InterpolationError::MissingControlPoints)?;

            if x.len() != y.len() {
                return Err(InterpolationError::MismatchedLengths { x_len: x.len(), y_len: y.len() }.into());
            }
            if x.is_empty() {
                return Err(InterpolationError::EmptyControlPoints.into());
            }

            let x_eval = Array1::from_iter((0..CURVE_RESOLUTION).map(|i| i as i32));
            let interpolated = interpolation::pchip_interpolate(&x, &y, &x_eval)?;
            final_curve = interpolated.mapv(|v| v.clamp(0, (CURVE_RESOLUTION - 1) as i32));
        }

        self.get_adjustment_set(mask_name)?.brightness_tone_curve = final_curve;
        Ok(())
    }
    pub fn set_oklch_hue_curve(
        &mut self,
        curve: Option<Array1<i32>>,
        control_points_x: Option<Array1<i32>>,
        control_points_y: Option<Array1<i32>>,
        mask_name: Option<&str>,
    ) -> Result<(), PhotoEditorError> {
        if curve.is_none() && control_points_x.is_none() {
            return Err(InterpolationError::MissingCurveOrControlPoints.into());
        }

        let final_curve: Array1<i32>;

        if let Some(c) = curve {
            if c.len() != CURVE_RESOLUTION {
                return Err(InterpolationError::InvalidCurveLength { expected: CURVE_RESOLUTION, actual: c.len() }.into());
            }
            final_curve = c;
        } else {
            let x = control_points_x.ok_or(InterpolationError::MissingControlPoints)?;
            let y = control_points_y.ok_or(InterpolationError::MissingControlPoints)?;

            if x.len() != y.len() {
                return Err(InterpolationError::MismatchedLengths { x_len: x.len(), y_len: y.len() }.into());
            }
            if x.is_empty() {
                return Err(InterpolationError::EmptyControlPoints.into());
            }

            let x_eval = Array1::from_iter((0..CURVE_RESOLUTION).map(|i| i as i32));
            let interpolated = interpolation::pchip_interpolate(&x, &y, &x_eval)?;
            final_curve = interpolated.mapv(|v| v.clamp(0, 65535));
        }

        self.get_adjustment_set(mask_name)?.hue_tone_curve = final_curve;
        Ok(())
    }
    pub fn set_oklch_saturation_curve(
        &mut self,
        curve: Option<Array1<i32>>,
        control_points_x: Option<Array1<i32>>,
        control_points_y: Option<Array1<i32>>,
        mask_name: Option<&str>,
    ) -> Result<(), PhotoEditorError> {
        if curve.is_none() && control_points_x.is_none() {
            return Err(InterpolationError::MissingCurveOrControlPoints.into());
        }

        let final_curve: Array1<i32>;

        if let Some(c) = curve {
            if c.len() != CURVE_RESOLUTION {
                return Err(InterpolationError::InvalidCurveLength { expected: CURVE_RESOLUTION, actual: c.len() }.into());
            }
            final_curve = c;
        } else {
            let x = control_points_x.ok_or(InterpolationError::MissingControlPoints)?;
            let y = control_points_y.ok_or(InterpolationError::MissingControlPoints)?;

            if x.len() != y.len() {
                return Err(InterpolationError::MismatchedLengths { x_len: x.len(), y_len: y.len() }.into());
            }
            if x.is_empty() {
                return Err(InterpolationError::EmptyControlPoints.into());
            }

            let x_eval = Array1::from_iter((0..CURVE_RESOLUTION).map(|i| i as i32));
            let interpolated = interpolation::pchip_interpolate(&x, &y, &x_eval)?;
            final_curve = interpolated.mapv(|v| (v * 2).clamp(0, 131070));
        }

        self.get_adjustment_set(mask_name)?.saturation_tone_curve = final_curve;
        Ok(())

    }
    pub fn set_oklch_lightness_curve(
        &mut self,
        curve: Option<Array1<i32>>,
        control_points_x: Option<Array1<i32>>,
        control_points_y: Option<Array1<i32>>,
        mask_name: Option<&str>,
    ) -> Result<(), PhotoEditorError> {
        if curve.is_none() && control_points_x.is_none() {
            return Err(InterpolationError::MissingCurveOrControlPoints.into());
        }

        let final_curve: Array1<i32>;

        if let Some(c) = curve {
            if c.len() != CURVE_RESOLUTION {
                return Err(InterpolationError::InvalidCurveLength { expected: CURVE_RESOLUTION, actual: c.len() }.into());
            }
            final_curve = c;
        } else {
            let x = control_points_x.ok_or(InterpolationError::MissingControlPoints)?;
            let y = control_points_y.ok_or(InterpolationError::MissingControlPoints)?;

            if x.len() != y.len() {
                return Err(InterpolationError::MismatchedLengths { x_len: x.len(), y_len: y.len() }.into());
            }
            if x.is_empty() {
                return Err(InterpolationError::EmptyControlPoints.into());
            }

            let x_eval = Array1::from_iter((0..CURVE_RESOLUTION).map(|i| i as i32));
            let interpolated = interpolation::pchip_interpolate(&x, &y, &x_eval)?;
            final_curve = interpolated.mapv(|v| (v * 2).clamp(0, 131070));
        }

        self.get_adjustment_set(mask_name)?.lightness_tone_curve = final_curve;
        Ok(())
    }

    pub fn add_mask(&mut self, name: &str, mask_data: Array2<f32>) {
        let mask_range = self.get_adjustment_set(None).unwrap().mask_range;
        let binarized_mask_data: Vec<f32> = mask_data.iter().map(|&v| if v >= mask_range { 1.0 } else { 0.0 }).collect();
        let gpu_mask = Self::create_gpu_mask(
            self.gpu_processor.device(),
            self.gpu_processor.queue(),
            self.original_image.width,
            self.original_image.height,
            Some(&binarized_mask_data)
        );
        self.masks.insert(name.to_string(), (gpu_mask, EditParameters::default()));
    }


    pub fn remove_mask(&mut self, name: &str) {
        if name != "main" {
            self.masks.remove(name);
        }
    }
    
    pub fn apply_adjustments(&mut self) -> Result<(), PhotoEditorError> {
        let processed_image = self.gpu_processor.apply_adjustments(
            &self.original_image,
            &self.masks,
        )?;

        self.image = processed_image;

        Ok(())
    }
}
