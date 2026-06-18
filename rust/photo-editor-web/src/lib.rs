use photo_editor::{GpuProcessor, ImageFormat, PhotoEditor};

use ndarray::Array1;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WebGpuProcessor {
    inner: Arc<GpuProcessor>,
}

#[wasm_bindgen]
pub struct WebPhotoEditor {
    inner: PhotoEditor,
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
impl WebGpuProcessor {
    #[wasm_bindgen]
    pub async fn create() -> Result<WebGpuProcessor, JsValue> {
        let gpu = GpuProcessor::new(0)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self {
            inner: Arc::new(gpu),
        })
    }
}

#[wasm_bindgen]
impl WebPhotoEditor {
    #[wasm_bindgen(constructor)]
    pub fn new(
        gpu: &WebGpuProcessor,
        image_bytes: &[u8],
        extension: &str,
    ) -> Result<WebPhotoEditor, JsValue> {
        println!("{}", extension);
        let format =
            ImageFormat::from_ext(extension).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let editor = PhotoEditor::new(gpu.inner.clone(), image_bytes, format)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        println!("{}", editor.image.height);
        Ok(Self { inner: editor })
    }

    pub fn create_from_rgb_f32(
        gpu: &WebGpuProcessor,
        data: Vec<f32>,
        width: u32,
        height: u32,
    ) -> Result<WebPhotoEditor, JsValue> {
        let editor =
            PhotoEditor::new_from_rgb_f32(gpu.inner.clone(), data, height as usize, width as usize)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self { inner: editor })
    }

    pub fn width(&self) -> u32 {
        self.inner.image.width
    }

    pub fn height(&self) -> u32 {
        self.inner.image.height
    }

    /// ホワイトバランスを設定
    pub fn set_whitebalance(
        &mut self,
        temperature: i32,
        tint: i32,
        mask_name: Option<String>,
    ) -> Result<(), JsValue> {
        self.inner
            .set_whitebalance(temperature, tint, mask_name.as_deref())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// ビネット（周辺減光）を設定
    pub fn set_vignette(&mut self, value: i32) -> Result<(), JsValue> {
        self.inner
            .set_vignette(value)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// レンズ歪み補正を設定
    pub fn set_lens_distortion_correction(&mut self, value: i32) -> Result<(), JsValue> {
        self.inner
            .set_lens_distortion_correction(value)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// 基本的なトーン調整を一括で設定
    pub fn set_tone(
        &mut self,
        exposure: f32,
        contrast: i32,
        shadow: i32,
        highlight: i32,
        black: i32,
        white: i32,
        mask_name: Option<String>,
    ) -> Result<(), JsValue> {
        self.inner
            .set_tone(
                exposure,
                contrast,
                shadow,
                highlight,
                black,
                white,
                mask_name.as_deref(),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// JS側の Vec<i32> を Rust の ndarray::Array1 に変換するヘルパー
    fn js_array_to_array1(js_array: Option<Vec<i32>>) -> Option<Array1<i32>> {
        js_array.map(|arr| Array1::from_vec(arr.to_vec()))
    }

    /// 明るさのトーンカーブを設定
    pub fn set_brightness_tone_curve(
        &mut self,
        curve: Option<Vec<i32>>,
        control_points_x: Option<Vec<i32>>,
        control_points_y: Option<Vec<i32>>,
        mask_name: Option<String>,
    ) -> Result<(), JsValue> {
        let c = Self::js_array_to_array1(curve);
        let cx = Self::js_array_to_array1(control_points_x);
        let cy = Self::js_array_to_array1(control_points_y);

        self.inner
            .set_brightness_tone_curve(c, cx, cy, mask_name.as_deref())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// 色相（Hue）のトーンカーブを設定
    pub fn set_oklch_hue_curve(
        &mut self,
        curve: Option<Vec<i32>>,
        control_points_x: Option<Vec<i32>>,
        control_points_y: Option<Vec<i32>>,
        mask_name: Option<String>,
    ) -> Result<(), JsValue> {
        let c = Self::js_array_to_array1(curve);
        let cx = Self::js_array_to_array1(control_points_x);
        let cy = Self::js_array_to_array1(control_points_y);

        self.inner
            .set_oklch_hue_curve(c, cx, cy, mask_name.as_deref())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// 彩度（Saturation）のトーンカーブを設定
    pub fn set_oklch_saturation_curve(
        &mut self,
        curve: Option<Vec<i32>>,
        control_points_x: Option<Vec<i32>>,
        control_points_y: Option<Vec<i32>>,
        mask_name: Option<String>,
    ) -> Result<(), JsValue> {
        let c = Self::js_array_to_array1(curve);
        let cx = Self::js_array_to_array1(control_points_x);
        let cy = Self::js_array_to_array1(control_points_y);

        self.inner
            .set_oklch_saturation_curve(c, cx, cy, mask_name.as_deref())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// 明度（Lightness）のトーンカーブを設定
    pub fn set_oklch_lightness_curve(
        &mut self,
        curve: Option<Vec<i32>>,
        control_points_x: Option<Vec<i32>>,
        control_points_y: Option<Vec<i32>>,
        mask_name: Option<String>,
    ) -> Result<(), JsValue> {
        let c = Self::js_array_to_array1(curve);
        let cx = Self::js_array_to_array1(control_points_x);
        let cy = Self::js_array_to_array1(control_points_y);

        self.inner
            .set_oklch_lightness_curve(c, cx, cy, mask_name.as_deref())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn apply(&mut self) -> Result<(), JsValue> {
        self.inner
            .apply_adjustments()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub async fn get_rgb_f32(&self) -> Result<Vec<f32>, JsValue> {
        self.inner
            .image
            .to_flat_vec_rgb()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub async fn get_rgba_f32(&self) -> Result<Vec<f32>, JsValue> {
        self.inner
            .image
            .to_flat_vec_rgba()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub async fn save_png(&self) -> Result<Vec<u8>, JsValue> {
        self.inner
            .save(&ImageFormat::PNG)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub async fn save_jpeg(&self) -> Result<Vec<u8>, JsValue> {
        self.inner
            .save(&ImageFormat::JPEG)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn exif_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner.get_exif_hashmap())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
