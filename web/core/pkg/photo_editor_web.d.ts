/* tslint:disable */
/* eslint-disable */

export class WebGpuProcessor {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static create(): Promise<WebGpuProcessor>;
}

export class WebPhotoEditor {
    free(): void;
    [Symbol.dispose](): void;
    add_mask(name: string, data: Float32Array, width: number, height: number): void;
    apply(): void;
    static create_from_rgb_f32(gpu: WebGpuProcessor, data: Float32Array, width: number, height: number): WebPhotoEditor;
    exif_json(): string;
    get_rgb_f32(): Promise<Float32Array>;
    get_rgba_f32(): Promise<Float32Array>;
    height(): number;
    constructor(gpu: WebGpuProcessor, image_bytes: Uint8Array, extension: string);
    remove_mask(name: string): void;
    save_jpeg(): Promise<Uint8Array>;
    save_png(): Promise<Uint8Array>;
    /**
     * 明るさのトーンカーブを設定
     */
    set_brightness_tone_curve(curve?: Int32Array | null, control_points_x?: Int32Array | null, control_points_y?: Int32Array | null, mask_name?: string | null): void;
    /**
     * レンズ歪み補正を設定
     */
    set_lens_distortion_correction(value: number): void;
    set_mask_range(mask_name: string, mask_range: number): void;
    /**
     * 色相（Hue）のトーンカーブを設定
     */
    set_oklch_hue_curve(curve?: Int32Array | null, control_points_x?: Int32Array | null, control_points_y?: Int32Array | null, mask_name?: string | null): void;
    /**
     * 明度（Lightness）のトーンカーブを設定
     */
    set_oklch_lightness_curve(curve?: Int32Array | null, control_points_x?: Int32Array | null, control_points_y?: Int32Array | null, mask_name?: string | null): void;
    /**
     * 彩度（Saturation）のトーンカーブを設定
     */
    set_oklch_saturation_curve(curve?: Int32Array | null, control_points_x?: Int32Array | null, control_points_y?: Int32Array | null, mask_name?: string | null): void;
    /**
     * 基本的なトーン調整を一括で設定
     */
    set_tone(exposure: number, contrast: number, shadow: number, highlight: number, black: number, white: number, mask_name?: string | null): void;
    /**
     * ビネット（周辺減光）を設定
     */
    set_vignette(value: number): void;
    /**
     * ホワイトバランスを設定
     */
    set_whitebalance(temperature: number, tint: number, mask_name?: string | null): void;
    width(): number;
}

export function init(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_webgpuprocessor_free: (a: number, b: number) => void;
    readonly __wbg_webphotoeditor_free: (a: number, b: number) => void;
    readonly webgpuprocessor_create: () => any;
    readonly webphotoeditor_add_mask: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly webphotoeditor_apply: (a: number) => [number, number];
    readonly webphotoeditor_create_from_rgb_f32: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly webphotoeditor_exif_json: (a: number) => [number, number, number, number];
    readonly webphotoeditor_get_rgb_f32: (a: number) => any;
    readonly webphotoeditor_get_rgba_f32: (a: number) => any;
    readonly webphotoeditor_height: (a: number) => number;
    readonly webphotoeditor_new: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly webphotoeditor_remove_mask: (a: number, b: number, c: number) => [number, number];
    readonly webphotoeditor_save_jpeg: (a: number) => any;
    readonly webphotoeditor_save_png: (a: number) => any;
    readonly webphotoeditor_set_brightness_tone_curve: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
    readonly webphotoeditor_set_lens_distortion_correction: (a: number, b: number) => [number, number];
    readonly webphotoeditor_set_mask_range: (a: number, b: number, c: number, d: number) => [number, number];
    readonly webphotoeditor_set_oklch_hue_curve: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
    readonly webphotoeditor_set_oklch_lightness_curve: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
    readonly webphotoeditor_set_oklch_saturation_curve: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
    readonly webphotoeditor_set_tone: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
    readonly webphotoeditor_set_vignette: (a: number, b: number) => [number, number];
    readonly webphotoeditor_set_whitebalance: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly webphotoeditor_width: (a: number) => number;
    readonly init: () => void;
    readonly wasm_bindgen_46a9236ac5bcbc21___convert__closures_____invoke___wasm_bindgen_46a9236ac5bcbc21___JsValue__core_f0fd674eaa06beef___result__Result_____wasm_bindgen_46a9236ac5bcbc21___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_46a9236ac5bcbc21___convert__closures_____invoke___js_sys_ed5f3a2b9f4611a2___Function_fn_wasm_bindgen_46a9236ac5bcbc21___JsValue_____wasm_bindgen_46a9236ac5bcbc21___sys__Undefined___js_sys_ed5f3a2b9f4611a2___Function_fn_wasm_bindgen_46a9236ac5bcbc21___JsValue_____wasm_bindgen_46a9236ac5bcbc21___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_46a9236ac5bcbc21___convert__closures_____invoke___wasm_bindgen_46a9236ac5bcbc21___JsValue______true_: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
