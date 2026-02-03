// image.rs

use crate::errors::PhotoEditorError;
use crate::metadata;
use exif::{Reader as ExifReader, Tag};
use image;
use ndarray::{Array3, ShapeError};
use rawler::{decoders, imgop, rawsource};
use std::io::Cursor;
use std::sync::Arc;

// 一般的な画像とRAW画像を入れる
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    PNG,
    JPEG,
    WEBP,
    TIFF,

    ARI,
    ARW,
    CR2,
    CR3,
    CRM,
    CRW,
    DCR,
    DCS,
    DNG,
    ERF,
    IIQ,
    KDC,
    MEF,
    MOS,
    MRW,
    NEF,
    NRW,
    ORF,
    ORI,
    PEF,
    RAF,
    RAW,
    RW2,
    RWL,
    SRW,
    _3FR,
    FFF,
    X3F,
    QTK,

    Unknown,
}

impl ImageFormat {
    pub fn from_ext(ext: &str) -> Result<ImageFormat, PhotoEditorError> {
        let image_format = match ext.to_lowercase().as_str() {
            "jpg" | "jpeg" => ImageFormat::JPEG,
            "png" => ImageFormat::PNG,
            "webp" => ImageFormat::WEBP,
            "tiff" | "tif" => ImageFormat::TIFF,

            "ari" => ImageFormat::ARI,
            "arw" => ImageFormat::ARW,
            "cr2" => ImageFormat::CR2,
            "cr3" => ImageFormat::CR3,
            "crm" => ImageFormat::CRM,
            "crw" => ImageFormat::CRW,
            "dcr" => ImageFormat::DCR,
            "dcs" => ImageFormat::DCS,
            "dng" => ImageFormat::DNG,
            "erf" => ImageFormat::ERF,
            "iiq" => ImageFormat::IIQ,
            "kdc" => ImageFormat::KDC,
            "mef" => ImageFormat::MEF,
            "mos" => ImageFormat::MOS,
            "mrw" => ImageFormat::MRW,
            "nef" => ImageFormat::NEF,
            "nrw" => ImageFormat::NRW,
            "orf" => ImageFormat::ORF,
            "ori" => ImageFormat::ORI,
            "pef" => ImageFormat::PEF,
            "raf" => ImageFormat::RAF,
            "raw" => ImageFormat::RAW,
            "rw2" => ImageFormat::RW2,
            "rwl" => ImageFormat::RWL,
            "srw" => ImageFormat::SRW,
            "3fr" => ImageFormat::_3FR,
            "fff" => ImageFormat::FFF,
            "x3f" => ImageFormat::X3F,
            "qtk" => ImageFormat::QTK,

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

            ImageFormat::ARI => "ari",
            ImageFormat::ARW => "arw",
            ImageFormat::CR2 => "cr2",
            ImageFormat::CR3 => "cr3",
            ImageFormat::CRM => "crm",
            ImageFormat::CRW => "crw",
            ImageFormat::DCR => "dcr",
            ImageFormat::DCS => "dcs",
            ImageFormat::DNG => "dng",
            ImageFormat::ERF => "erf",
            ImageFormat::IIQ => "iiq",
            ImageFormat::KDC => "kdc",
            ImageFormat::MEF => "mef",
            ImageFormat::MOS => "mos",
            ImageFormat::MRW => "mrw",
            ImageFormat::NEF => "nef",
            ImageFormat::NRW => "nrw",
            ImageFormat::ORF => "orf",
            ImageFormat::ORI => "ori",
            ImageFormat::PEF => "pef",
            ImageFormat::RAF => "raf",
            ImageFormat::RAW => "raw",
            ImageFormat::RW2 => "rw2",
            ImageFormat::RWL => "rwl",
            ImageFormat::SRW => "srw",
            ImageFormat::_3FR => "3fr",
            ImageFormat::FFF => "fff",
            ImageFormat::X3F => "x3f",
            ImageFormat::QTK => "qtk",

            ImageFormat::Unknown => "unknown",
        }
    }

    pub fn is_standard_image(&self) -> bool {
        match self {
            ImageFormat::JPEG | ImageFormat::PNG | ImageFormat::WEBP | ImageFormat::TIFF => true,
            _ => false,
        }
    }

    pub fn is_raw_image(&self) -> bool {
        match self {
            ImageFormat::ARI
            | ImageFormat::ARW
            | ImageFormat::CR2
            | ImageFormat::CR3
            | ImageFormat::CRM
            | ImageFormat::CRW
            | ImageFormat::DCR
            | ImageFormat::DCS
            | ImageFormat::DNG
            | ImageFormat::ERF
            | ImageFormat::IIQ
            | ImageFormat::KDC
            | ImageFormat::MEF
            | ImageFormat::MOS
            | ImageFormat::MRW
            | ImageFormat::NEF
            | ImageFormat::NRW
            | ImageFormat::ORF
            | ImageFormat::ORI
            | ImageFormat::PEF
            | ImageFormat::RAF
            | ImageFormat::RAW
            | ImageFormat::RW2
            | ImageFormat::RWL
            | ImageFormat::SRW
            | ImageFormat::_3FR
            | ImageFormat::FFF
            | ImageFormat::X3F
            | ImageFormat::QTK => true,

            _ => false,
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

    pub fn to_flat_vec(&self) -> Result<Vec<f32>, PhotoEditorError> {
        let texture_size = self.texture.size();

        // RGBA (4) * f32 (4 bytes)
        const BYTES_PER_PIXEL: u32 = 16;
        // wgpu requires that bytes_per_row be a multiple of 256
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
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(PhotoEditorError::gpu_compute)?;

        rx.recv()
            .map_err(PhotoEditorError::gpu_compute)?
            .map_err(PhotoEditorError::gpu_compute)?;

        let data = buffer_slice.get_mapped_range();
        let mut rgb_data = Vec::with_capacity((self.width * self.height * 3) as usize);

        for row_bytes in data.chunks(aligned_bytes_per_row as usize) {
            let data_row = &row_bytes[..unaligned_bytes_per_row as usize];
            let row_f32: &[f32] = bytemuck::cast_slice(data_row);

            // Convert RGBA to RGB
            rgb_data.extend(row_f32.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]));
        }

        Ok(rgb_data)
    }

    pub fn to_array3(&self) -> Result<Array3<f32>, PhotoEditorError> {
        let flat_vec = self.to_flat_vec()?;
        Ok(
            Array3::from_shape_vec((self.height as usize, self.width as usize, 3), flat_vec)
                .expect("Failed to create Array3 from flat vec"),
        )
    }

    pub fn to_array3_u8(&self) -> Result<Array3<u8>, PhotoEditorError> {
        Ok(self
            .to_array3()?
            .mapv(|v| (v.clamp(0.0, 1.0) * 255.0) as u8))
    }

    pub fn to_u8_rgbimage(&self) -> Result<image::RgbImage, PhotoEditorError> {
        let flat_vec = self.to_flat_vec()?;
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
    if image_format.is_standard_image() {
        Ok(read_standard_image(device, queue, file_data, image_format)?)
    } else if image_format.is_raw_image() {
        Ok(read_raw_image(device, queue, file_data, image_format)?)
    } else {
        Err(PhotoEditorError::ReadImageUnsupportedFormat(
            image_format.to_str().to_string(),
        ))
    }
}

fn read_standard_image(
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
        _ => panic!("Non-standard format passed to read_standard_image"),
    };

    let file_cursor = Cursor::new(&file_data);
    let dynamic_image =
        image::load(file_cursor, image_crate_format).map_err(PhotoEditorError::standard_image)?;

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

fn read_raw_image(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    file_data: &[u8],
    _image_format: &ImageFormat,
) -> Result<(Image, metadata::Exif), PhotoEditorError> {
    // read pixels
    let raw_source = rawsource::RawSource::new_from_slice(&file_data);
    let raw_decoder = rawler::get_decoder(&raw_source).map_err(PhotoEditorError::raw_image)?;
    let raw_image = raw_decoder
        .raw_image(&raw_source, &decoders::RawDecodeParams::default(), false)
        .map_err(PhotoEditorError::raw_image)?;
    let raw_metadata = raw_decoder
        .raw_metadata(&raw_source, &decoders::RawDecodeParams::default())
        .map_err(PhotoEditorError::raw_image)?;

    let mut dynamic_image = imgop::develop::RawDevelop::default()
        .develop_intermediate(&raw_image)
        .map_err(PhotoEditorError::raw_image)?
        .to_dynamic_image()
        .unwrap();

    let exif = raw_metadata.exif;
    dynamic_image = apply_exif_orientation(dynamic_image, exif.orientation);

    let width = dynamic_image.width();
    let height = dynamic_image.height();
    let rgb_image = dynamic_image.to_rgb32f();
    let image_vec = rgb_image.into_raw();

    let image = Image::new(device, queue, &image_vec, width, height);

    // read exif
    let mut exif_data = metadata::Exif::default();

    exif_data.datetime = exif.date_time_original;
    exif_data.f_number = exif.fnumber.map(|v| v.as_f32());
    exif_data.flash = exif.flash.map(|v| v.to_string());
    exif_data.exposure_time = exif.exposure_time.map(|v| v.to_string());
    exif_data.focal_length = exif.focal_length.map(|v| v.as_f32() as u32);
    exif_data.iso = exif.iso_speed_ratings.map(|v| v as u32);
    exif_data.lens_make = exif.lens_make;
    exif_data.lens_model = exif.lens_model;
    exif_data.model = Some(raw_metadata.model);
    exif_data.make = Some(raw_metadata.make);
    exif_data.exposure_bias = exif.exposure_bias.map(|v| v.n as f32 / v.d as f32);

    Ok((image, exif_data))
}

fn apply_exif_orientation(
    mut dynamic_image: image::DynamicImage,
    orientation: Option<u16>,
) -> image::DynamicImage {
    match orientation {
        Some(o) => {
            match o {
                1 => {
                    // そのまま（通常）
                }
                2 => {
                    // 左右反転
                    dynamic_image = dynamic_image.fliph();
                }
                3 => {
                    // 180度回転
                    dynamic_image = dynamic_image.rotate180();
                }
                4 => {
                    // 上下反転
                    dynamic_image = dynamic_image.flipv();
                }
                5 => {
                    // 上下反転＋270度回転
                    dynamic_image = dynamic_image.flipv().rotate270();
                }
                6 => {
                    // 右90度回転
                    dynamic_image = dynamic_image.rotate90();
                }
                7 => {
                    // 左右反転＋90度回転
                    dynamic_image = dynamic_image.fliph().rotate90();
                }
                8 => {
                    // 左90度回転
                    dynamic_image = dynamic_image.rotate270();
                }
                _ => {
                    // 不明値は無視
                }
            }
        }
        None => {
            // Orientation情報なし → そのまま
        }
    }

    dynamic_image
}

pub fn write_image(image: &Image, image_format: &ImageFormat) -> Result<Vec<u8>, PhotoEditorError> {
    if !image_format.is_standard_image() {
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
        .to_u8_rgbimage()?
        .write_to(&mut writer, image_crate_format)
        .map_err(PhotoEditorError::save)?;

    Ok(writer.into_inner())
}
