// metadata.rs

use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct Exif {
    pub datetime: Option<String>,
    pub f_number: Option<f32>,
    pub flash: Option<String>,
    pub lens_make: Option<String>,
    pub lens_model: Option<String>,
    pub model: Option<String>,
    pub make: Option<String>,
    pub focal_length: Option<u32>,
    pub exposure_time: Option<String>,
    pub iso: Option<u32>,
    pub exposure_bias: Option<f32>,

}

impl Exif {
    pub fn to_hashmap(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();

        if let Some(v) = &self.datetime {
            map.insert("DateTimeOriginal".into(), v.clone());
        }
        if let Some(v) = self.f_number {
            map.insert("FNumber".into(), v.to_string());
        }
        if let Some(v) = &self.exposure_time {
            map.insert("ExposureTime".into(), v.clone());
        }
        if let Some(v) = self.iso {
            map.insert("ISO".into(), v.to_string());
        }
        if let Some(v) = self.exposure_bias {
            map.insert("ExposureBiasValue".into(), v.to_string());
        }
        if let Some(v) = self.focal_length {
            map.insert("FocalLength".into(), v.to_string());
        }
        if let Some(v) = &self.make {
            map.insert("Make".into(), v.clone());
        }
        if let Some(v) = &self.model {
            map.insert("Model".into(), v.clone());
        }
        if let Some(v) = &self.lens_make {
            map.insert("LensMake".into(), v.clone());
        }
        if let Some(v) = &self.lens_model {
            map.insert("LensModel".into(), v.clone());
        }
        if let Some(v) = &self.flash {
            map.insert("Flash".into(), v.clone());
        }

        map
    }
}

