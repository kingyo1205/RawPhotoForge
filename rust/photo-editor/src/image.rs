// image.rs


use std::io::{Cursor};
use image;
use ndarray::{Array3, ShapeError};
use rawler::{decoders, rawsource, imgop};
use exif::{Reader as ExifReader, Tag};

use crate::errors::PhotoEditorError;
use crate::metadata;

// 一般的な画像とRAW画像を入れる
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    PNG, JPEG, WEBP, TIFF,

    ARI, ARW, CR2, CR3, CRM, CRW, DCR, DCS, DNG, ERF, IIQ, KDC, MEF, MOS, MRW, NEF, NRW, ORF, ORI, PEF, RAF, RAW, RW2, RWL, SRW, _3FR, FFF, X3F, QTK,

    Unknown

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

            _ => ImageFormat::Unknown
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

            ImageFormat::Unknown => "unknown"
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
            ImageFormat::ARI |
            ImageFormat::ARW |
            ImageFormat::CR2 |
            ImageFormat::CR3 | 
            ImageFormat::CRM | 
            ImageFormat::CRW | 
            ImageFormat::DCR | 
            ImageFormat::DCS | 
            ImageFormat::DNG | 
            ImageFormat::ERF | 
            ImageFormat::IIQ |
            ImageFormat::KDC |
            ImageFormat::MEF | 
            ImageFormat::MOS | 
            ImageFormat::MRW | 
            ImageFormat::NEF | 
            ImageFormat::NRW | 
            ImageFormat::ORF | 
            ImageFormat::ORI | 
            ImageFormat::PEF |
            ImageFormat::RAF |
            ImageFormat::RAW |
            ImageFormat::RW2 |
            ImageFormat::RWL |
            ImageFormat::SRW |
            ImageFormat::_3FR |
            ImageFormat::FFF | 
            ImageFormat::X3F | 
            ImageFormat::QTK  => true,

            _ => false
       }
    }
    
}

#[derive(Clone)]
pub struct Image {
    pub data: Array3<f32>,
    pub height: usize,
    pub width: usize
}


impl Image {


    pub fn new(data: Array3<f32>, height: usize, width: usize) -> Image {
        Image {
            data,
            height,
            width
        }
    }

    pub fn new_from_vec(image_vec: Vec<f32>, height: usize, width: usize) -> Result<Image, ShapeError> {
        let image = Image {
            data: Array3::from_shape_vec((height, width, 3), image_vec)?,
            height,
            width
        };
        Ok(image)
    }


    pub fn to_flat_vec(&self) -> Vec<f32> {
        self.data.iter().map(|v| v.clone()).collect()
    }


    pub fn to_array3(&self) -> Array3<f32> {
        self.data.clone()
    }

    pub fn to_array3_u8(&self) -> Array3<u8> {
        let mut array3 = Array3::zeros((self.height, self.width, 3));
        for y in 0..self.height {
            for x in 0..self.width {
                let r = (self.data[[y, x, 0]].clamp(0.0, 1.0) * 255.0) as u8;
                let g = (self.data[[y, x, 1]].clamp(0.0, 1.0) * 255.0) as u8;
                let b = (self.data[[y, x, 2]].clamp(0.0, 1.0) * 255.0) as u8;
                array3[[y, x, 0]] = r;
                array3[[y, x, 1]] = g;
                array3[[y, x, 2]] = b;
            }
        }
        array3
    }

    pub fn to_u8_rgbimage(&self) -> image::RgbImage {
        let (h, w, c) = self.data.dim();
        assert_eq!(c, 3, "チャンネルは3(RGB)である必要がある");

        let mut img = image::RgbImage::new(w as u32, h as u32);

        for y in 0..h {
            for x in 0..w {
                let r = (self.data[[y, x, 0]].clamp(0.0, 1.0) * 255.0) as u8;
                let g = (self.data[[y, x, 1]].clamp(0.0, 1.0) * 255.0) as u8;
                let b = (self.data[[y, x, 2]].clamp(0.0, 1.0) * 255.0) as u8;
                img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
            }
        }

        img
    }
}








pub fn read_image(file_data: &[u8], image_format: &ImageFormat) -> Result<(Image, metadata::Exif), PhotoEditorError> {

    if image_format.is_standard_image() {
        Ok(read_standard_image(file_data, image_format)?)
    } else if image_format.is_raw_image() {
        Ok(read_raw_image(file_data, image_format)?)
    } else {
        Err(PhotoEditorError::ReadImageUnsupportedFormat(image_format.to_str().to_string()))
    }
    
}


fn read_standard_image(file_data: &[u8], image_format: &ImageFormat) -> Result<(Image, metadata::Exif), PhotoEditorError> {

    let image_crate_format = match image_format {
        ImageFormat::JPEG => image::ImageFormat::Jpeg,
        ImageFormat::PNG => image::ImageFormat::Png,
        ImageFormat::WEBP => image::ImageFormat::WebP,
        ImageFormat::TIFF => image::ImageFormat::Tiff,
        _ => panic!("Non-standard format passed to read_standard_image"),
    };

    // read pixels
    let file_cursor = Cursor::new(&file_data);
    let dynamic_image = image::load(file_cursor, image_crate_format).map_err(PhotoEditorError::standard_image)?;

    let height = dynamic_image.height() as usize;
    let width = dynamic_image.width() as usize;

    let image_vec = dynamic_image.to_rgb8().to_vec();

    drop(dynamic_image);

    let image_vec = image_vec.iter().map(|v| (v.clone() as f32) / 255.0).collect::<Vec<f32>>();
    
    let image = Image::new_from_vec(image_vec, height, width).map_err(PhotoEditorError::standard_image)?;


    // read exif
    let mut exif_data = metadata::Exif::default();
    let mut file_cursor  = Cursor::new(&file_data);
    if let Ok(exif) = ExifReader::new().read_from_container(&mut file_cursor) {
        for f in exif.fields() {
            match f.tag {
                Tag::DateTimeOriginal => exif_data.datetime = Some(f.display_value().to_string()),
                Tag::FNumber => exif_data.f_number = f.display_value().to_string().parse::<f32>().ok(),
                Tag::Flash=> exif_data.flash = Some(f.display_value().to_string()),
                Tag::LensMake => exif_data.lens_make = Some(f.display_value().to_string()),
                Tag::LensModel => exif_data.lens_model = Some(f.display_value().to_string()),
                Tag::Model => exif_data.model = Some(f.display_value().to_string()),
                Tag::Make => exif_data.make = Some(f.display_value().to_string()),
                Tag::FocalLength => exif_data.focal_length = f.display_value().to_string().parse::<u32>().ok(),
                Tag::ExposureTime => exif_data.exposure_time = Some(f.display_value().to_string()),
                Tag::ISOSpeed => exif_data.iso = f.display_value().to_string().parse::<u32>().ok(),
                Tag::ExposureBiasValue => exif_data.exposure_bias = f.display_value().to_string().parse::<f32>().ok(),
                Tag::PhotographicSensitivity => exif_data.iso = f.display_value().to_string().parse::<u32>().ok(),
                _ => {}
            }
        }
    }      
    
    Ok((image, exif_data))
}




fn read_raw_image(file_data: &[u8], _image_format: &ImageFormat) -> Result<(Image, metadata::Exif), PhotoEditorError> {

    // read pixels
    let raw_source = rawsource::RawSource::new_from_slice(&file_data);
    let raw_decoder = rawler::get_decoder(&raw_source).map_err(PhotoEditorError::raw_image)?;
    let raw_image = raw_decoder.raw_image(&raw_source, &decoders::RawDecodeParams::default(), false).map_err(PhotoEditorError::raw_image)?;
    let raw_metadata = raw_decoder.raw_metadata(&raw_source, &decoders::RawDecodeParams::default()).map_err(PhotoEditorError::raw_image)?;

    let mut dynamic_image = imgop::develop::RawDevelop::default().develop_intermediate(&raw_image).map_err(PhotoEditorError::raw_image)?.to_dynamic_image().unwrap();

    let exif = raw_metadata.exif;
    dynamic_image = apply_exif_orientation(dynamic_image, exif.orientation);

    let height = dynamic_image.height() as usize;
    let width = dynamic_image.width() as usize;

    let image_vec = dynamic_image.to_rgb32f().to_vec();

    let image = Image::new_from_vec(image_vec, height, width).map_err(PhotoEditorError::raw_image)?;

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






fn apply_exif_orientation(mut dynamic_image: image::DynamicImage, orientation: Option<u16>) -> image::DynamicImage {
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
        return Err(PhotoEditorError::SaveImageUnsupportedFormat(image_format.to_str().to_string()));
    }

    let image_crate_format = match image_format {
        ImageFormat::JPEG => image::ImageFormat::Jpeg,
        ImageFormat::PNG => image::ImageFormat::Png,
        ImageFormat::WEBP => image::ImageFormat::WebP,
        ImageFormat::TIFF => image::ImageFormat::Tiff,
        _ => panic!("Failed to convert to image crate format: {}", image_format.to_str()),
    };

    let mut writer = Cursor::new(Vec::<u8>::new());
    image.to_u8_rgbimage().write_to(&mut writer, image_crate_format).map_err(PhotoEditorError::save)?;

    Ok(writer.into_inner())
}