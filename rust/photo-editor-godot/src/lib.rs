use godot::prelude::*;
use godot::builtin::{PackedByteArray, PackedVector2Array, VarDictionary}; // Removed PackedInt32Array
use photo_editor::{self, PhotoEditor as PhotoEditorImpl};
use photo_editor::image::ImageFormat as ImageFormatImpl;
use ndarray::{Array1, Array2};
use anyhow::{Result};

/// photo-editorクレートのImageFormat enumのGodotラッパー。
/// Godotエディタ上で文字列として選択できます。
#[derive(Debug, Clone, Copy, PartialEq, Eq, GodotConvert, Var, Export)]
#[godot(via = GString)]
pub enum ImageFormat {
    Png, Jpeg, Webp, Tiff,
    ARI, ARW, CR2, CR3, CRM, CRW, DCR, DCS, DNG, ERF, IIQ, KDC, MEF, MOS, MRW, NEF, NRW, ORF, ORI, PEF, RAF, RAW, RW2, RWL, SRW, _3FR, FFF, X3F, QTK
}

// photo-editorのImageFormatとの相互変換
impl From<ImageFormat> for ImageFormatImpl {
    fn from(format: ImageFormat) -> Self {
        match format {
            ImageFormat::Png => Self::Png, ImageFormat::Jpeg => Self::Jpeg, ImageFormat::Webp => Self::Webp, ImageFormat::Tiff => Self::Tiff,
            ImageFormat::ARI => Self::ARI, ImageFormat::ARW => Self::ARW, ImageFormat::CR2 => Self::CR2, ImageFormat::CR3 => Self::CR3,
            ImageFormat::CRM => Self::CRM, ImageFormat::CRW => Self::CRW, ImageFormat::DCR => Self::DCR, ImageFormat::DCS => Self::DCS,
            ImageFormat::DNG => Self::DNG, ImageFormat::ERF => Self::ERF, ImageFormat::IIQ => Self::IIQ, ImageFormat::KDC => Self::KDC,
            ImageFormat::MEF => Self::MEF, ImageFormat::MOS => Self::MOS, ImageFormat::MRW => Self::MRW, ImageFormat::NEF => Self::NEF,
            ImageFormat::NRW => Self::NRW, ImageFormat::ORF => Self::ORF, ImageFormat::ORI => Self::ORI, ImageFormat::PEF => Self::PEF,
            ImageFormat::RAF => Self::RAF, ImageFormat::RAW => Self::RAW, ImageFormat::RW2 => Self::RW2, ImageFormat::RWL => Self::RWL,
            ImageFormat::SRW => Self::SRW, ImageFormat::_3FR => Self::_3FR, ImageFormat::FFF => Self::FFF, ImageFormat::X3F => Self::X3F,
            ImageFormat::QTK => Self::QTK,
        }
    }
}

/// photo_editorクレートのPhotoEditor構造体のGodotラッパー
#[derive(GodotClass)]
#[class(base=Node, init)]
pub struct PhotoEditor {
    base: Base<Node>,
    editor: Option<PhotoEditorImpl>,
}

#[godot_api]
impl PhotoEditor {
    fn base(&self) -> &Base<Node> {
        &self.base
    }

    fn base_mut(&mut self) -> &mut Base<Node> {
        &mut self.base
    }

    fn init(base: Base<Node>) -> Self {
        Self {
            base,
            editor: None,
        }
    }

    /// 画像データを渡して新しいPhotoEditorインスタンスを作成します。
    #[func]
    pub fn open_image(&mut self, file_data: PackedByteArray, format_str: GString) -> bool {
        let format_str = format_str.to_string();
        let format = match ImageFormatImpl::from_ext(&format_str) {
            Ok(f) => f,
            Err(e) => {
                godot_error!("Unsupported format: {}, error: {}", format_str, e);
                return false;
            }
        };

        let data_vec = file_data.to_vec();
        match PhotoEditorImpl::new(data_vec.as_slice(), format) {
            Ok(editor) => {
                self.editor = Some(editor);
                true
            }
            Err(e) => {
                godot_error!("Failed to create PhotoEditor: {}", e);
                false
            }
        }
    }

    /// 現在の編集済み画像をGodotのImageオブジェクトとして返します。
    #[func]
    pub fn get_image(&self) -> Variant {
        if let Some(editor) = &self.editor {
            let image_impl = &editor.image;
            let width = image_impl.width as i32;
            let height = image_impl.height as i32;
            let image_data_f32 = image_impl.to_flat_vec();

            let mut raw_bytes: Vec<u8> = Vec::with_capacity((width * height * 3 * 4) as usize);
            for &f in &image_data_f32 {
                raw_bytes.extend_from_slice(&f.to_le_bytes());
            }
            let byte_array = PackedByteArray::from(raw_bytes);

            let image = godot::classes::Image::create_from_data(width, height, false, godot::classes::image::Format::RGBF, &byte_array);
            
            Variant::from(image)
        } else {
            godot_warn!("Editor not initialized");
            Variant::nil()
        }
    }

    /// EXIF情報をDictionaryとして返します。
    #[func]
    pub fn get_exif(&self) -> VarDictionary {
        if let Some(editor) = &self.editor {
            let mut dict = VarDictionary::new();
            for (key, value) in editor.get_exif_hashmap() {
                dict.set(GString::from(key.as_str()), GString::from(value.as_str()));
            }
            dict
        } else {
            godot_warn!("Editor not initialized");
            VarDictionary::new()
        }
    }

    /// 編集を適用します。
    #[func]
    pub fn apply_adjustments(&mut self) -> bool {
        if let Some(editor) = self.editor.as_mut() {
            match editor.apply_adjustments() {
                Ok(_) => true,
                Err(e) => {
                    godot_error!("Failed to apply adjustments: {}", e);
                    false
                }
            }
        } else {
            godot_warn!("Editor not initialized");
            false
        }
    }

    /// 指定されたフォーマットで画像を保存し、バイト配列を返します。
    #[func]
    pub fn save(&self, format: ImageFormat) -> Variant {
        if let Some(editor) = &self.editor {
            match editor.save(&format.into()) {
                Ok(data) => Variant::from(PackedByteArray::from(data)),
                Err(e) => {
                    godot_error!("Failed to save image: {}", e);
                    Variant::nil()
                }
            }
        } else {
            godot_warn!("Editor not initialized");
            Variant::nil()
        }
    }

    /// 全ての編集をリセットして元の画像に戻します。
    #[func]
    pub fn reset(&mut self) {
        if let Some(editor) = self.editor.as_mut() {
            editor.reset();
        } else {
            godot_warn!("Editor not initialized");
        }
    }

    /// ホワイトバランスを設定します。
    #[func]
    pub fn set_whitebalance(&mut self, temperature: i32, tint: i32, mask_name: GString) {
        if let Some(editor) = self.editor.as_mut() {
            let mask = if mask_name.is_empty() { None } else { Some(mask_name.to_string()) };
            if let Err(e) = editor.set_whitebalance(temperature, tint, mask.as_deref()) {
                godot_error!("Failed to set white balance: {}", e);
            }
        } else {
            godot_warn!("Editor not initialized");
        }
    }

    /// ヴィネット効果を設定します。
    #[func]
    pub fn set_vignette(&mut self, value: i32) {
        if let Some(editor) = self.editor.as_mut() {
            if let Err(e) = editor.set_vignette(value) {
                godot_error!("Failed to set vignette: {}", e);
            }
        } else {
            godot_warn!("Editor not initialized");
        }
    }

    /// レンズ歪み補正を設定します。
    #[func]
    pub fn set_lens_distortion_correction(&mut self, value: i32) {
        if let Some(editor) = self.editor.as_mut() {
            if let Err(e) = editor.set_lens_distortion_correction(value) {
                godot_error!("Failed to set lens distortion: {}", e);
            }
        } else {
            godot_warn!("Editor not initialized");
        }
    }
    
    /// 明るさ関連の調整を設定します。
    #[func]
    pub fn set_tone(&mut self, exposure: f32, contrast: i32, shadow: i32, highlight: i32, black: i32, white: i32, mask_name: GString) {
        if let Some(editor) = self.editor.as_mut() {
            let mask = if mask_name.is_empty() { None } else { Some(mask_name.to_string()) };
            if let Err(e) = editor.set_tone(exposure, contrast, shadow, highlight, black, white, mask.as_deref()) {
                 godot_error!("Failed to set tone: {}", e);
            }
        } else {
            godot_warn!("Editor not initialized");
        }
    }

    fn set_curve_from_points(
        editor: &mut PhotoEditorImpl,
        points: PackedVector2Array,
        mask_name: GString,
        setter: impl Fn(&mut PhotoEditorImpl, Option<Array1<i32>>, Option<Array1<i32>>, Option<Array1<i32>>, Option<&str>) -> Result<()>,
    ) {
        let points_vec = points.to_vec();
        let mut x = Vec::new();
        let mut y = Vec::new();
        for p in points_vec {
            x.push(p.x as i32);
            y.push(p.y as i32);
        }
        let x_arr = Some(Array1::from_vec(x));
        let y_arr = Some(Array1::from_vec(y));
        let mask = if mask_name.is_empty() { None } else { Some(mask_name.to_string()) };

        if let Err(e) = setter(editor, None, x_arr, y_arr, mask.as_deref()) {
            godot_error!("Failed to set curve from points: {}", e);
        }
    }
    
    #[func]
    pub fn set_brightness_tone_curve_from_points(&mut self, points: PackedVector2Array, mask_name: GString) {
        if let Some(editor) = self.editor.as_mut() {
            Self::set_curve_from_points(editor, points, mask_name, |e, _, x, y, m| Ok(e.set_brightness_tone_curve(None, x, y, m)?));
        } else {
            godot_warn!("Editor not initialized");
        }
    }

    #[func]
    pub fn set_oklch_hue_curve_from_points(&mut self, points: PackedVector2Array, mask_name: GString) {
        if let Some(editor) = self.editor.as_mut() {
            Self::set_curve_from_points(editor, points, mask_name, |e, _, x, y, m| Ok(e.set_oklch_hue_curve(None, x, y, m)?));
        } else {
            godot_warn!("Editor not initialized");
        }
    }

    #[func]
    pub fn set_oklch_saturation_curve_from_points(&mut self, points: PackedVector2Array, mask_name: GString) {
        if let Some(editor) = self.editor.as_mut() {
            Self::set_curve_from_points(editor, points, mask_name, |e, _, x, y, m| Ok(e.set_oklch_saturation_curve(None, x, y, m)?));
        } else {
            godot_warn!("Editor not initialized");
        }
    }

    #[func]
    pub fn set_oklch_lightness_curve_from_points(&mut self, points: PackedVector2Array, mask_name: GString) {
        if let Some(editor) = self.editor.as_mut() {
            Self::set_curve_from_points(editor, points, mask_name, |e, _, x, y, m| Ok(e.set_oklch_lightness_curve(None, x, y, m)?));
        } else {
            godot_warn!("Editor not initialized");
        }
    }

    #[func]
    pub fn add_mask_from_image(&mut self, name: GString, mask_image: Gd<godot::classes::Image>) {
        if let Some(editor) = self.editor.as_mut() {
            let width = mask_image.get_width() as usize;
            let height = mask_image.get_height() as usize;

            if width != editor.image.width || height != editor.image.height {
                godot_error!("Mask dimensions ({}, {}) must match image dimensions ({}, {})", width, height, editor.image.width, editor.image.height);
                return;
            }
            
            let data = mask_image.get_data();
            let data_vec = data.to_vec();
            
            let mask_data: Vec<f32> = match mask_image.get_format() {
                 godot::classes::image::Format::L8 => {
                     data_vec.iter().map(|&val| val as f32 / 255.0).collect()
                 },
                 godot::classes::image::Format::RGB8 => {
                      data_vec.iter().step_by(3).map(|&r| r as f32 / 255.0).collect()
                 },
                 godot::classes::image::Format::RGBA8 => {
                      data_vec.iter().step_by(4).map(|&r| r as f32 / 255.0).collect()
                 },
                 _ => {
                     godot_error!("Unsupported mask image format. Please use L8, RGB8, or RGBA8.");
                     return;
                 }
            };
            
            let mask_array = Array2::from_shape_vec((height, width), mask_data).unwrap();
            editor.add_mask(&name.to_string(), mask_array);

        } else {
            godot_warn!("Editor not initialized");
        }
    }

    #[func]
    pub fn remove_mask(&mut self, name: GString) {
        if let Some(editor) = self.editor.as_mut() {
            editor.remove_mask(&name.to_string());
        } else {
            godot_warn!("Editor not initialized");
        }
    }
}

// Godot extension entry point
struct PhotoEditorGodotLibrary;

#[gdextension]
unsafe impl ExtensionLibrary for PhotoEditorGodotLibrary {}
