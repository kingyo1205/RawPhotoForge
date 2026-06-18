// image.rs

use crate::errors::PhotoEditorError;
use crate::metadata;
use exif::{Reader as ExifReader, Tag};
use futures::channel::oneshot;
use image::ImageDecoder;
use ndarray::{Array3, ShapeError};
use std::io::Cursor;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    PNG,
    JPEG,
    WEBP,
    TIFF,

    Unknown,
}

impl ImageFormat {
    pub fn from_ext(ext: &str) -> Result<ImageFormat, PhotoEditorError> {
        let image_format = match ext.to_lowercase().as_str() {
            "jpg" | "jpeg" => ImageFormat::JPEG,
            "png" => ImageFormat::PNG,
            "webp" => ImageFormat::WEBP,
            "tiff" | "tif" => ImageFormat::TIFF,

            _ => ImageFormat::Unknown,
        };

        Ok(image_format)
    }

    pub fn to_str(&self) -> &str {
        match self {
            ImageFormat::JPEG => "jpeg",
            ImageFormat::PNG => "png",
            ImageFormat::WEBP => "webp",
            ImageFormat::TIFF => "tiff",

            ImageFormat::Unknown => "unknown",
        }
    }
}

pub struct Image {
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl Clone for Image {
    fn clone(&self) -> Self {
        let new_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cloned Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Clone Encoder"),
            });

        encoder.copy_texture_to_texture(
            self.texture.as_image_copy(),
            new_texture.as_image_copy(),
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));
        let texture_view = new_texture.create_view(&Default::default());

        Self {
            texture: new_texture,
            texture_view: texture_view,
            width: self.width,
            height: self.height,
            device: self.device.clone(),
            queue: self.queue.clone(),
        }
    }
}

impl Image {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        rgb_data: &[f32],
        width: u32,
        height: u32,
    ) -> Image {
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Image Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        if !rgb_data.is_empty() {
            let rgba_data: Vec<f32> = rgb_data
                .chunks_exact(3)
                .flat_map(|c| [c[0], c[1], c[2], 1.0])
                .collect();

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&rgba_data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * 4 * width),
                    rows_per_image: Some(height),
                },
                texture_size,
            );
        }

        let texture_view = texture.create_view(&Default::default());
        Image {
            texture,
            texture_view,
            width,
            height,
            device,
            queue,
        }
    }

    pub fn from_texture(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        texture: wgpu::Texture,
        texture_view: wgpu::TextureView,
    ) -> Image {
        let (width, height) = (texture.width(), texture.height());
        Image {
            texture,
            texture_view,
            width: width,
            height: height,
            device,
            queue,
        }
    }

    pub fn new_from_vec(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        image_vec: Vec<f32>,
        height: usize,
        width: usize,
    ) -> Result<Image, ShapeError> {
        Ok(Image::new(
            device,
            queue,
            &image_vec,
            width as u32,
            height as u32,
        ))
    }

    pub async fn to_flat_vec_rgb(&self) -> Result<Vec<f32>, PhotoEditorError> {
        let texture_size = self.texture.size();

        const BYTES_PER_PIXEL: u32 = 16;
        const ALIGNMENT: u32 = 256;

        let unaligned_bytes_per_row = BYTES_PER_PIXEL * self.width;
        let aligned_bytes_per_row = (unaligned_bytes_per_row + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        let buffer_size = (aligned_bytes_per_row * self.height) as u64;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Image Readback Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback Encoder"),
            });

        encoder.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            texture_size,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = output_buffer.slice(..);

        let (tx, rx) = oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(PhotoEditorError::gpu_compute)?;

        rx.await
            .map_err(PhotoEditorError::gpu_compute)?
            .map_err(PhotoEditorError::gpu_compute)?;

        let data = buffer_slice.get_mapped_range();

        let mut rgb_data = Vec::with_capacity((self.width * self.height * 3) as usize);

        for row_bytes in data.chunks(aligned_bytes_per_row as usize) {
            let data_row = &row_bytes[..unaligned_bytes_per_row as usize];

            let row_f32: &[f32] = bytemuck::cast_slice(data_row);

            rgb_data.extend(row_f32.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]));
        }

        drop(data);
        output_buffer.unmap();

        Ok(rgb_data)
    }

    pub async fn to_flat_vec_rgba(&self) -> Result<Vec<f32>, PhotoEditorError> {
        let texture_size = self.texture.size();

        const BYTES_PER_PIXEL: u32 = 16;
        const ALIGNMENT: u32 = 256;

        let unaligned_bytes_per_row = BYTES_PER_PIXEL * self.width;
        let aligned_bytes_per_row = (unaligned_bytes_per_row + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        let buffer_size = (aligned_bytes_per_row * self.height) as u64;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Image Readback Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback Encoder"),
            });

        encoder.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            texture_size,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = output_buffer.slice(..);

        let (tx, rx) = oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(PhotoEditorError::gpu_compute)?;

        rx.await
            .map_err(PhotoEditorError::gpu_compute)?
            .map_err(PhotoEditorError::gpu_compute)?;

        let data = buffer_slice.get_mapped_range();

        let mut rgba_data = Vec::with_capacity((self.width * self.height * 4) as usize);

        for row_bytes in data.chunks(aligned_bytes_per_row as usize) {
            let data_row = &row_bytes[..unaligned_bytes_per_row as usize];

            let row_f32: &[f32] = bytemuck::cast_slice(data_row);

            rgba_data.extend(
                row_f32
                    .chunks_exact(4)
                    .flat_map(|c| [c[0], c[1], c[2], c[3]]),
            );
        }

        drop(data);
        output_buffer.unmap();

        Ok(rgba_data)
    }

    pub async fn to_rgb_array3(&self) -> Result<Array3<f32>, PhotoEditorError> {
        let flat_vec = self.to_flat_vec_rgb().await?;
        Ok(
            Array3::from_shape_vec((self.height as usize, self.width as usize, 3), flat_vec)
                .expect("Failed to create Array3 from flat vec"),
        )
    }

    pub async fn to_rgb_array3_u8(&self) -> Result<Array3<u8>, PhotoEditorError> {
        Ok(self
            .to_rgb_array3()
            .await?
            .mapv(|v| (v.clamp(0.0, 1.0) * 255.0) as u8))
    }

    pub async fn to_u8_rgbimage(&self) -> Result<image::RgbImage, PhotoEditorError> {
        let flat_vec = self.to_flat_vec_rgb().await?;
        let u8_vec: Vec<u8> = flat_vec
            .into_iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();
        Ok(image::RgbImage::from_raw(self.width, self.height, u8_vec)
            .expect("Failed to create RgbImage from raw vec"))
    }
}

pub fn read_image(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    file_data: &[u8],
    image_format: &ImageFormat,
) -> Result<(Image, metadata::Exif), PhotoEditorError> {
    if let ImageFormat::Unknown = image_format {
        Err(PhotoEditorError::ReadImageUnsupportedFormat(
            image_format.to_str().to_string(),
        ))
    } else {
        Ok(read_image_and_exif(device, queue, file_data, image_format)?)
    }
}

fn read_image_and_exif(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    file_data: &[u8],
    image_format: &ImageFormat,
) -> Result<(Image, metadata::Exif), PhotoEditorError> {
    let image_crate_format = match image_format {
        ImageFormat::JPEG => image::ImageFormat::Jpeg,
        ImageFormat::PNG => image::ImageFormat::Png,
        ImageFormat::WEBP => image::ImageFormat::WebP,
        ImageFormat::TIFF => image::ImageFormat::Tiff,
        _ => panic!("Non-supported format passed to read_image_and_exif"),
    };

    let file_cursor = Cursor::new(&file_data);

    let mut decoder = image::ImageReader::with_format(file_cursor, image_crate_format)
        .into_decoder()
        .map_err(PhotoEditorError::image)?;

    let orientation = decoder.orientation().map_err(PhotoEditorError::image)?;

    let mut dynamic_image =
        image::DynamicImage::from_decoder(decoder).map_err(PhotoEditorError::image)?;

    dynamic_image.apply_orientation(orientation);

    // println!("{:?}", dynamic_image.color_space());

    match image_format {
        ImageFormat::TIFF => {}
        _ => {
            dynamic_image
                .apply_color_space(
                    image::metadata::Cicp::SRGB_LINEAR,
                    image::ConvertColorOptions::default(),
                )
                .map_err(PhotoEditorError::image)?;
        }
    };
    let width = dynamic_image.width();
    let height = dynamic_image.height();
    let rgb_image = dynamic_image.to_rgb32f();
    let image_vec = rgb_image.into_raw();

    let image = Image::new(device, queue, &image_vec, width, height);

    // read exif
    let mut exif_data = metadata::Exif::default();
    let mut file_cursor = Cursor::new(&file_data);
    if let Ok(exif) = ExifReader::new().read_from_container(&mut file_cursor) {
        for f in exif.fields() {
            match f.tag {
                Tag::DateTimeOriginal => exif_data.datetime = Some(f.display_value().to_string()),
                Tag::FNumber => {
                    exif_data.f_number = f.display_value().to_string().parse::<f32>().ok()
                }
                Tag::Flash => exif_data.flash = Some(f.display_value().to_string()),
                Tag::LensMake => exif_data.lens_make = Some(f.display_value().to_string()),
                Tag::LensModel => exif_data.lens_model = Some(f.display_value().to_string()),
                Tag::Model => exif_data.model = Some(f.display_value().to_string()),
                Tag::Make => exif_data.make = Some(f.display_value().to_string()),
                Tag::FocalLength => {
                    exif_data.focal_length = f.display_value().to_string().parse::<u32>().ok()
                }
                Tag::ExposureTime => exif_data.exposure_time = Some(f.display_value().to_string()),
                Tag::ISOSpeed => exif_data.iso = f.display_value().to_string().parse::<u32>().ok(),
                Tag::ExposureBiasValue => {
                    exif_data.exposure_bias = f.display_value().to_string().parse::<f32>().ok()
                }
                Tag::PhotographicSensitivity => {
                    exif_data.iso = f.display_value().to_string().parse::<u32>().ok()
                }
                _ => {}
            }
        }
    }

    Ok((image, exif_data))
}

pub async fn write_image(
    image: &Image,
    image_format: &ImageFormat,
) -> Result<Vec<u8>, PhotoEditorError> {
    if let ImageFormat::Unknown = image_format {
        return Err(PhotoEditorError::SaveImageUnsupportedFormat(
            image_format.to_str().to_string(),
        ));
    }

    let image_crate_format = match image_format {
        ImageFormat::JPEG => image::ImageFormat::Jpeg,
        ImageFormat::PNG => image::ImageFormat::Png,
        ImageFormat::WEBP => image::ImageFormat::WebP,
        ImageFormat::TIFF => image::ImageFormat::Tiff,
        _ => panic!(
            "Failed to convert to image crate format: {}",
            image_format.to_str()
        ),
    };

    let mut writer = Cursor::new(Vec::<u8>::new());
    image
        .to_u8_rgbimage()
        .await?
        .write_to(&mut writer, image_crate_format)
        .map_err(PhotoEditorError::save)?;

    Ok(writer.into_inner())
}
