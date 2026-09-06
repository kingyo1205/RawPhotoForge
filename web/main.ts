import init, {
    WebGpuProcessor as GpuImageProcessor,
    WebPhotoEditor as PhotoEditor,
} from "photo-editor-web";
import { RawImage, ensureModelLoaded, generateMask } from './ai_mask';
import { CurveMode, ToneCurveEditor } from './tone_curve_editor';
import translation from "./translations/translation.json?raw";
await init();

type Translations = Record<string, Record<string, string>>;

interface Settings {
    uiPreviewSize: number;
    dragPreviewSize: number;
    locale: string;
}

interface Float32ArrayImageRGB {
    data: Float32Array<ArrayBuffer>;
    width: number;
    height: number;
}

interface Float32Mask {
    data: Float32Array;
    width: number;
    height: number;
}

const defaultSettings: Settings = {
    uiPreviewSize: 1280,
    dragPreviewSize: 400,
    locale: 'en',
};

let settings: Settings = { ...defaultSettings };
const SETTINGS_FILE_PATH = "raw-photo-forge-settings";

class I18n {
    private lang: string;
    private data: Translations;

    constructor(data: Translations, lang: string) {
        this.data = data;
        this.lang = lang;
    }

    t(key: string): string {
        return this.data[this.lang]?.[key]
            ?? this.data["en"]?.[key]
            ?? key;
    }

    setLang(lang: string) {
        this.lang = lang;
    }
}

export interface EditParameters {
    exposure: number;
    contrast: number;
    shadow: number;
    highlight: number;
    black: number;
    white: number;
    temperature: number;
    tint: number;
    vignette: number;
    lens_distortion: number;
    brightness_tone_curve_points: { x: number, y: number }[];
    hue_tone_curve_points: { x: number, y: number }[];
    saturation_tone_curve_points: { x: number, y: number }[];
    lightness_tone_curve_points: { x: number, y: number }[];
    mask_range: number;
}

// 「現在UIが編集している対象」= 選択中マスクの edit_parameters への参照（エイリアス）。
// masks配列の該当エントリと同じオブジェクトを指すので、書き換えるとそのままmasksに反映される。
type EditState = EditParameters;

export interface Mask {
    name: string;
    edit_parameters: EditParameters;
    data: Float32Array | null; // mainはnull（生データ、二値化前）
}

function createDefaultParameters(): EditParameters {
    return {
        exposure: 0.0,
        contrast: 0,
        shadow: 0,
        highlight: 0,
        black: 0,
        white: 0,
        temperature: 0,
        tint: 0,
        vignette: 0,
        lens_distortion: 0,
        brightness_tone_curve_points: [{ x: 0.0, y: 0.0 }, { x: 1.0, y: 1.0 }],
        hue_tone_curve_points: [{ x: 0.0, y: 0.0 }, { x: 1.0, y: 1.0 }],
        saturation_tone_curve_points: [{ x: 0.0, y: 1.0 }, { x: 1.0, y: 1.0 }],
        lightness_tone_curve_points: [{ x: 0.0, y: 1.0 }, { x: 1.0, y: 1.0 }],
        mask_range: 0.0,
    };
}

let masks: Mask[] = [{
    name: "main",
    edit_parameters: createDefaultParameters(),
    data: null
}];

enum PreviewLevel { LOW, MID, FULL }

// --- AIマスク関連の状態 ---
let maskCounter = 1;
let selectedMaskName = "main";
let isCreatingMask = false;
let clickPoints: { x: number, y: number }[] = [];
let showMask = false;

const maskDataCache = new Map<string, Float32Mask>();      // フル解像度の生データ（二値化前）
const maskOverlayCache = new Map<string, Float32Mask>();   // プレビュー表示用リサイズキャッシュ（生データ）

let modelLoadPromise: Promise<void> | null = null;
let captureImagePromise: Promise<InstanceType<typeof RawImage>> | null = null;

let gpuProcessor: GpuImageProcessor;
let editorFull: PhotoEditor | null = null;
let editorMid: PhotoEditor | null = null;
let editorLow: PhotoEditor | null = null;

let uploadTexture: GPUTexture | null = null;

let currentImageFile: File | null = null;
let imageLoaded = false;
let previewLevel = PreviewLevel.MID;

let uniformBuffer: GPUBuffer;

let toneCurveEditors: { [key: string]: ToneCurveEditor } = {};

// UIが今編集している対象（選択中マスクへのエイリアス）
let editState: EditState = masks[0].edit_parameters;
// vignette / lens_distortion はマスク非対応のためmain固定で参照する
let mainEditParams: EditParameters = masks[0].edit_parameters;

let canvasContext: GPUCanvasContext | null = null;
let presentationFormat: GPUTextureFormat;
let renderPipeline: GPURenderPipeline | null = null;
let isRendering = false;

const translationsData: Translations = JSON.parse(translation);
const i18n: I18n = new I18n(translationsData, "en");

class WebGpuContext {
    adapter: GPUAdapter;
    device: GPUDevice;
    queue: GPUQueue;

    static async create(): Promise<WebGpuContext> {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error("No GPU adapter");
        }
        const device = await adapter.requestDevice();
        return {
            adapter,
            device,
            queue: device.queue,
        };
    }
}

let gpu: WebGpuContext;

const renderShaderCode = `
    @vertex
    fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
        let xy = array<vec2<f32>, 4>(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>(1.0, -1.0),
            vec2<f32>(-1.0, 1.0),
            vec2<f32>(1.0, 1.0)
        );
        return vec4<f32>(xy[in_vertex_index], 0.0, 1.0);
    }

    @group(0) @binding(0) var imgTexture: texture_2d<f32>;

    struct Uniforms {
        canvasSize: vec2<f32>,
        textureSize: vec2<f32>,
    };

    @group(0) @binding(1) var<uniform> u: Uniforms;

    @fragment
    fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
        let uv = pos.xy / u.canvasSize;
        let tex_coord = vec2<i32>(uv * u.textureSize);
        return textureLoad(imgTexture, tex_coord, 0);
    }
`;

const ui = {
    mainCanvas: document.getElementById('main-canvas') as HTMLCanvasElement,
    fileInput: document.getElementById('file-input') as HTMLInputElement,

    exposureLabel: document.getElementById('exposure-label') as HTMLLabelElement,
    contrastLabel: document.getElementById('contrast-label') as HTMLLabelElement,
    shadowLabel: document.getElementById('shadow-label') as HTMLLabelElement,
    highlightLabel: document.getElementById('highlight-label') as HTMLLabelElement,
    blackLabel: document.getElementById('black-label') as HTMLLabelElement,
    whiteLabel: document.getElementById('white-label') as HTMLLabelElement,
    temperatureLabel: document.getElementById('temperature-label') as HTMLLabelElement,
    tintLabel: document.getElementById('tint-label') as HTMLLabelElement,
    vignetteLabel: document.getElementById('vignette-label') as HTMLLabelElement,
    lensDistortionLabel: document.getElementById('lens-distortion-label') as HTMLLabelElement,

    exposureSlider: document.getElementById('exposure-slider') as HTMLInputElement,
    contrastSlider: document.getElementById('contrast-slider') as HTMLInputElement,
    shadowSlider: document.getElementById('shadow-slider') as HTMLInputElement,
    highlightSlider: document.getElementById('highlight-slider') as HTMLInputElement,
    blackSlider: document.getElementById('black-slider') as HTMLInputElement,
    whiteSlider: document.getElementById('white-slider') as HTMLInputElement,
    temperatureSlider: document.getElementById('temperature-slider') as HTMLInputElement,
    tintSlider: document.getElementById('tint-slider') as HTMLInputElement,
    vignetteSlider: document.getElementById('vignette-slider') as HTMLInputElement,
    lensDistortionSlider: document.getElementById('lens-distortion-slider') as HTMLInputElement,

    resetToneButton: document.getElementById('reset-tone-button') as HTMLButtonElement,
    resetWbButton: document.getElementById('reset-wb-button') as HTMLButtonElement,
    resetEffectButton: document.getElementById('reset-effect-button') as HTMLButtonElement,
    resetBrightnessButton: document.getElementById('reset-brightness-button') as HTMLButtonElement,
    resetHueButton: document.getElementById('reset-hue-button') as HTMLButtonElement,
    resetSaturationButton: document.getElementById('reset-saturation-button') as HTMLButtonElement,
    resetLightnessButton: document.getElementById('reset-lightness-button') as HTMLButtonElement,

    tabButtons: document.querySelectorAll('.tab-button'),
    tabPanes: document.querySelectorAll('.tab-pane'),

    openFile: document.getElementById('open-file') as HTMLDivElement,
    saveFile: document.getElementById('save-file') as HTMLDivElement,
    resetAll: document.getElementById('reset-all') as HTMLDivElement,

    saveDialog: document.getElementById('save-dialog') as HTMLDivElement,
    saveDialogSave: document.getElementById('save-dialog-save') as HTMLButtonElement,
    saveDialogCancel: document.getElementById('save-dialog-cancel') as HTMLButtonElement,
    formatSelect: document.getElementById('format-select') as HTMLSelectElement,

    settingsMenu: document.getElementById('settings-menu') as HTMLDivElement,
    settingsDialog: document.getElementById('settings-dialog') as HTMLDivElement,
    settingsDialogSave: document.getElementById('settings-dialog-save') as HTMLButtonElement,
    settingsDialogCancel: document.getElementById('settings-dialog-cancel') as HTMLButtonElement,
    uiPreviewSizeSlider: document.getElementById('ui-preview-size-slider') as HTMLInputElement,
    uiPreviewSizeInput: document.getElementById('ui-preview-size-input') as HTMLInputElement,
    dragPreviewSizeSlider: document.getElementById('drag-preview-size-slider') as HTMLInputElement,
    dragPreviewSizeInput: document.getElementById('drag-preview-size-input') as HTMLInputElement,
    languageSelect: document.getElementById('language-select') as HTMLSelectElement,

    infoDialog: document.getElementById('info-dialog') as HTMLDivElement,
    infoDialogText: document.getElementById('info-dialog-text') as HTMLParagraphElement,
    infoDialogOk: document.getElementById('info-dialog-ok') as HTMLButtonElement,

    aiMaskStatusLabel: document.getElementById("ai-mask-status") as HTMLLabelElement,
    maskSelect: document.getElementById("mask-select") as HTMLSelectElement,
    btnCreateMask: document.getElementById("btn-create-mask") as HTMLButtonElement,
    actionContainer: document.getElementById("ai-mask-action-container") as HTMLDivElement,
    btnExecInference: document.getElementById("btn-exec-inference") as HTMLButtonElement,
    btnCancelMask: document.getElementById("btn-cancel-mask") as HTMLButtonElement,
    btnDeleteMask: document.getElementById("btn-delete-mask") as HTMLButtonElement,
    btnInvertMask: document.getElementById("btn-invert-mask") as HTMLButtonElement,
    chkShowMask: document.getElementById("chk-show-mask") as HTMLInputElement,
    maskRangeSlider: document.getElementById("mask-range-slider") as HTMLInputElement,
    maskRangeLabel: document.getElementById("mask-range-label") as HTMLLabelElement,
};

function applyI18n(i18n: I18n) {
    document.querySelectorAll<HTMLElement>("[data-i18n]").forEach(el => {
        el.textContent = i18n.t(el.dataset.i18n!);
    });
    document.querySelectorAll<HTMLInputElement>("[data-i18n-placeholder]").forEach(el => {
        el.placeholder = i18n.t(el.dataset.i18nPlaceholder!);
    });
    document.querySelectorAll<HTMLImageElement>("[data-i18n-alt]").forEach(el => {
        el.alt = i18n.t(el.dataset.i18nAlt!);
    });
}

function showInfoDialog(message: string) {
    ui.infoDialogText.textContent = message;
    ui.infoDialog.style.display = 'flex';
}

function loadSettings() {
    const savedSettings = localStorage.getItem(SETTINGS_FILE_PATH);
    if (savedSettings) {
        try {
            const parsed = JSON.parse(savedSettings);
            settings = { ...defaultSettings, ...parsed };
        } catch (e) {
            console.error("Failed to parse settings, using defaults.", e);
            settings = { ...defaultSettings };
        }
    } else {
        const browserLang = navigator.language.split('-')[0];
        if (browserLang === 'ja') {
            defaultSettings.locale = 'ja';
        }
        settings = { ...defaultSettings };
    }
}

function saveSettings() {
    settings.uiPreviewSize = parseInt(ui.uiPreviewSizeInput.value, 10);
    settings.dragPreviewSize = parseInt(ui.dragPreviewSizeInput.value, 10);
    settings.locale = ui.languageSelect.value;

    try {
        localStorage.setItem(SETTINGS_FILE_PATH, JSON.stringify(settings));
        return true;
    } catch (e) {
        console.error("Failed to save settings.", e);
        return false;
    }
}

function applySettings() {
    i18n.setLang(settings.locale);
    applyI18n(i18n);
    updateAllSliderLabels();
}

function updateSettingsUI() {
    ui.uiPreviewSizeSlider.value = String(settings.uiPreviewSize);
    ui.uiPreviewSizeInput.value = String(settings.uiPreviewSize);
    ui.dragPreviewSizeSlider.value = String(settings.dragPreviewSize);
    ui.dragPreviewSizeInput.value = String(settings.dragPreviewSize);
    ui.languageSelect.value = settings.locale;
}

async function initializeApp() {
    loadSettings();
    applySettings();
    setupEventListeners();
    setupAiMaskEventListeners();
    setupToneCurveEditors();
    updateAllSliderLabels();
    updateMaskControlsEnabled();

    const observer = new MutationObserver((mutations) => {
        for (const m of mutations) {
            m.addedNodes.forEach(node => {
                if (node instanceof HTMLElement) {
                    if (node.dataset.i18n) {
                        node.textContent = i18n.t(node.dataset.i18n);
                    }
                    node.querySelectorAll?.("[data-i18n]").forEach(el => {
                        el.textContent = i18n.t((el as HTMLElement).dataset.i18n!);
                    });
                }
            });
        }
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    try {
        gpuProcessor = await GpuImageProcessor.create();
        console.log('WebGPU initialized successfully.');
    } catch (error) {
        console.error('WebGPU initialization failed:', error);
        alert(`${i18n.t("TR_ERROR_WEBGPU")}。${error}`);
        return;
    }

    gpu = await WebGpuContext.create();

    canvasContext = ui.mainCanvas.getContext("webgpu");
    if (!canvasContext) {
        alert(i18n.t("TR_ERROR_GET_WEBGPU_CANVAS_CONTEXT"));
        return;
    }
    presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    canvasContext.configure({
        device: gpu.device,
        format: presentationFormat,
        alphaMode: 'premultiplied',
    });

    const device = gpu.device;
    const shaderModule = device.createShaderModule({
        label: "Render Shader Module",
        code: renderShaderCode,
    });

    const renderBindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: 'unfilterable-float' },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: 'uniform' },
            },
        ],
    });

    uniformBuffer = device.createBuffer({
        size: 4 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    renderPipeline = device.createRenderPipeline({
        label: "Render to Canvas Pipeline",
        layout: device.createPipelineLayout({
            bindGroupLayouts: [renderBindGroupLayout],
        }),
        vertex: {
            module: shaderModule,
            entryPoint: "vs_main",
        },
        fragment: {
            module: shaderModule,
            entryPoint: "fs_main",
            targets: [{ format: presentationFormat }],
        },
        primitive: {
            topology: "triangle-strip",
            stripIndexFormat: "uint32",
        },
    });
}

async function uploadFloat32Texture(
    rgba: Float32Array,
    width: number,
    height: number
): Promise<GPUTextureView> {

    const device = gpu.device;

    if (
        !uploadTexture ||
        uploadTexture.width !== width ||
        uploadTexture.height !== height
    ) {
        uploadTexture?.destroy();

        uploadTexture = device.createTexture({
            size: [width, height],
            format: "rgba32float",
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST,
        });
    }

    device.queue.writeTexture(
        { texture: uploadTexture },
        rgba,
        {
            offset: 0,
            bytesPerRow: width * 16,
            rowsPerImage: height,
        },
        {
            width,
            height,
            depthOrArrayLayers: 1,
        }
    );

    return uploadTexture.createView();
}

function setupEventListeners() {

    ui.openFile.addEventListener('click', () => ui.fileInput.click());
    ui.fileInput.addEventListener('change', (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (file) {
            currentImageFile = file;
            loadImage(file);
        }
    });
    ui.saveFile.addEventListener('click', () => {
        if (!imageLoaded) return;
        ui.saveDialog.style.display = 'flex';
    });
    ui.resetAll.addEventListener('click', resetAllEdits);

    ui.saveDialogCancel.addEventListener('click', () => ui.saveDialog.style.display = 'none');
    ui.saveDialogSave.addEventListener('click', saveImage);

    ui.tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = (button as HTMLElement).dataset.tab as string;
            ui.tabButtons.forEach(btn => btn.classList.remove('active'));
            ui.tabPanes.forEach(pane => pane.classList.remove('active'));
            button.classList.add('active');
            const newPane = document.getElementById(`tab-${tabName}`);
            if (newPane) {
                newPane.classList.add('active');
            }

            if (toneCurveEditors[tabName]) {
                toneCurveEditors[tabName].draw();
            }
        });
    });

    // マスクごとに独立して持つパラメータ（editState = 選択中マスクのedit_parameters）
    const perMaskSliders = [
        { s: ui.exposureSlider, k: 'exposure', l: ui.exposureLabel, n: i18n.t("TR_EXPOSURE"), f: (v: number) => v.toFixed(2) },
        { s: ui.contrastSlider, k: 'contrast', l: ui.contrastLabel, n: i18n.t("TR_CONTRAST"), f: (v: number) => Math.round(v) },
        { s: ui.shadowSlider, k: 'shadow', l: ui.shadowLabel, n: i18n.t("TR_SHADOW"), f: (v: number) => Math.round(v) },
        { s: ui.highlightSlider, k: 'highlight', l: ui.highlightLabel, n: i18n.t("TR_HIGHLIGHT"), f: (v: number) => Math.round(v) },
        { s: ui.blackSlider, k: 'black', l: ui.blackLabel, n: i18n.t("TR_BLACK_LEVEL"), f: (v: number) => Math.round(v) },
        { s: ui.whiteSlider, k: 'white', l: ui.whiteLabel, n: i18n.t("TR_WHITE_LEVEL"), f: (v: number) => Math.round(v) },
        { s: ui.temperatureSlider, k: 'temperature', l: ui.temperatureLabel, n: i18n.t("TR_TEMPERATURE"), f: (v: number) => Math.round(v) },
        { s: ui.tintSlider, k: 'tint', l: ui.tintLabel, n: i18n.t("TR_TINT"), f: (v: number) => Math.round(v) },
    ];

    perMaskSliders.forEach(({ s, k, l, n, f }) => {
        s.addEventListener('input', () => {
            const value = parseFloat(s.value);
            (editState as any)[k] = value;
            l.textContent = `${n} ${f(value)}`;
            updateImage();
        });
        s.addEventListener('mousedown', onDragStart);
        s.addEventListener('mouseup', onDragEnd);
    });

    // マスク非対応（グローバルのみ）＝ 常にmainのパラメータを編集
    const globalSliders = [
        { s: ui.vignetteSlider, k: 'vignette', l: ui.vignetteLabel, n: i18n.t("TR_VIGNETTE"), f: (v: number) => Math.round(v) },
        { s: ui.lensDistortionSlider, k: 'lens_distortion', l: ui.lensDistortionLabel, n: i18n.t("TR_LENS_DISTORTION"), f: (v: number) => Math.round(v) },
    ];

    globalSliders.forEach(({ s, k, l, n, f }) => {
        s.addEventListener('input', () => {
            const value = parseFloat(s.value);
            (mainEditParams as any)[k] = value;
            l.textContent = `${n} ${f(value)}`;
            updateImage();
        });
        s.addEventListener('mousedown', onDragStart);
        s.addEventListener('mouseup', onDragEnd);
    });

    ui.resetToneButton.addEventListener('click', resetTone);
    ui.resetWbButton.addEventListener('click', resetWb);
    ui.resetEffectButton.addEventListener('click', resetEffect);
    ui.resetBrightnessButton.addEventListener('click', () => resetCurve('brightness'));
    ui.resetHueButton.addEventListener('click', () => resetCurve('hue'));
    ui.resetSaturationButton.addEventListener('click', () => resetCurve('saturation'));
    ui.resetLightnessButton.addEventListener('click', () => resetCurve('lightness'));

    ui.settingsMenu.addEventListener('click', () => {
        updateSettingsUI();
        ui.settingsDialog.style.display = 'flex';
    });
    ui.settingsDialogCancel.addEventListener('click', () => {
        ui.settingsDialog.style.display = 'none';
    });
    ui.settingsDialogSave.addEventListener('click', () => {
        if (saveSettings()) {
            applySettings();
            ui.settingsDialog.style.display = 'none';
            showInfoDialog(i18n.t("TR_SETTINGS_SAVED_INFO"));
            if (currentImageFile) {
                loadImage(currentImageFile);
            }
        }
    });

    ui.infoDialogOk.addEventListener('click', () => {
        ui.infoDialog.style.display = 'none';
    });

    ui.uiPreviewSizeSlider.addEventListener('input', () => {
        ui.uiPreviewSizeInput.value = ui.uiPreviewSizeSlider.value;
    });
    ui.uiPreviewSizeInput.addEventListener('change', () => {
        let value = parseInt(ui.uiPreviewSizeInput.value, 10);
        const min = parseInt(ui.uiPreviewSizeSlider.min, 10);
        const max = parseInt(ui.uiPreviewSizeSlider.max, 10);
        if (isNaN(value) || value < min) value = min;
        if (value > max) value = max;
        ui.uiPreviewSizeInput.value = String(value);
        ui.uiPreviewSizeSlider.value = String(value);
    });

    ui.dragPreviewSizeSlider.addEventListener('input', () => {
        ui.dragPreviewSizeInput.value = ui.dragPreviewSizeSlider.value;
    });
    ui.dragPreviewSizeInput.addEventListener('change', () => {
        let value = parseInt(ui.dragPreviewSizeInput.value, 10);
        const min = parseInt(ui.dragPreviewSizeSlider.min, 10);
        const max = parseInt(ui.dragPreviewSizeSlider.max, 10);
        if (isNaN(value) || value < min) value = min;
        if (value > max) value = max;
        ui.dragPreviewSizeInput.value = String(value);
        ui.dragPreviewSizeSlider.value = String(value);
    });
}

// ============================================================
// AIマスク機能
// ============================================================

function getMaskEntry(name: string): Mask | undefined {
    return masks.find(m => m.name === name);
}

function setupAiMaskEventListeners() {
    ui.maskSelect.addEventListener("change", (e) => {
        const name = (e.target as HTMLSelectElement).value;
        switchToMaskEditState(name);
    });

    ui.maskRangeSlider.addEventListener("input", (e) => {
        if (selectedMaskName === "main") return; // mainには意味がない
        const value = parseFloat((e.target as HTMLInputElement).value);
        editState.mask_range = value;
        updateMaskRangeLabel();
        updateImage();
    });
    ui.maskRangeSlider.addEventListener('mousedown', onDragStart);
    ui.maskRangeSlider.addEventListener('mouseup', onDragEnd);

    ui.chkShowMask.addEventListener("change", (e) => {
        showMask = (e.target as HTMLInputElement).checked;
        updateImage();
    });

    ui.btnCreateMask.addEventListener("click", () => {
        if (!imageLoaded || !editorFull) {
            showInfoDialog(i18n.t("TR_ERROR_NO_IMAGE_FOR_MASK"));
            return;
        }

        isCreatingMask = true;
        clickPoints = [];
        clearPointMarkers();
        ui.btnCreateMask.style.display = "none";
        ui.actionContainer.style.display = "flex";
        ui.mainCanvas.style.cursor = "crosshair";

        // クリックしている間にモデルのロードと画像取得を裏で進めておく
        modelLoadPromise = ensureModelLoaded((p) => { ui.aiMaskStatusLabel.textContent = `${i18n.t("TR_MASK_MODEL_LOAD_STATUS")}: ${p}%` });
        captureImagePromise = captureFullResRawImage();
    });

    ui.btnCancelMask.addEventListener("click", () => {
        exitMaskCreationMode();
    });

    ui.mainCanvas.addEventListener("click", (e) => {
        if (!isCreatingMask) return;

        const rect = ui.mainCanvas.getBoundingClientRect();
        const xFrac = (e.clientX - rect.left) / rect.width;
        const yFrac = (e.clientY - rect.top) / rect.height;

        if (xFrac < 0 || xFrac > 1 || yFrac < 0 || yFrac > 1) return;

        clickPoints.push({ x: xFrac, y: yFrac });
        addPointMarker(xFrac, yFrac);
    });

    ui.btnExecInference.addEventListener("click", async () => {
        if (!isCreatingMask) return;

        if (clickPoints.length === 0) {
            showInfoDialog(i18n.t("TR_ERROR_NO_MASK_POINTS"));
            return;
        }

        const points = clickPoints;
        exitMaskCreationMode();
        ui.btnExecInference.disabled = true;

        try {
            await modelLoadPromise;
            if (!captureImagePromise) throw new Error("No captured image for mask generation.");
            const rawImage = await captureImagePromise;

            const result = await generateMask(rawImage, points);

            // 二値化はしない。生データ（ロジット）のままキャッシュ・登録する。
            registerNewMask(result.data, result.width, result.height);
        } catch (err) {
            console.error("AI mask generation failed:", err);
            showInfoDialog(i18n.t("TR_ERROR_AI_MASK"));
        } finally {
            ui.btnExecInference.disabled = false;
            modelLoadPromise = null;
            captureImagePromise = null;
        }
    });

    ui.btnDeleteMask.addEventListener("click", () => {
        if (selectedMaskName === "main") return;

        const nameToDelete = selectedMaskName;

        removeMaskFromAllEditors(nameToDelete);
        maskDataCache.delete(nameToDelete);
        maskOverlayCache.delete(nameToDelete);
        masks = masks.filter(m => m.name !== nameToDelete);

        const optionToRemove = ui.maskSelect.querySelector(`option[value="${CSS.escape(nameToDelete)}"]`);
        optionToRemove?.remove();

        ui.maskSelect.value = "main";
        switchToMaskEditState("main");
    });

    ui.btnInvertMask.addEventListener("click", () => {
        if (selectedMaskName === "main") return;

        const original = maskDataCache.get(selectedMaskName);
        if (!original) return;

        // 生データ（ロジット）の符号を反転することで前景/背景を入れ替える
        const inverted = new Float32Array(original.data.length);
        for (let i = 0; i < original.data.length; i++) {
            inverted[i] = -original.data[i];
        }

        registerNewMask(inverted, original.width, original.height);
    });
}

function exitMaskCreationMode() {
    isCreatingMask = false;
    clickPoints = [];
    clearPointMarkers();
    ui.btnCreateMask.style.display = "block";
    ui.actionContainer.style.display = "none";
    ui.mainCanvas.style.cursor = "default";
}

function addPointMarker(xFrac: number, yFrac: number) {
    const container = ui.mainCanvas.parentElement;
    if (!container) return;

    const canvasRect = ui.mainCanvas.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    const marker = document.createElement("div");
    marker.className = "mask-point-marker";
    marker.style.left = `${canvasRect.left - containerRect.left + xFrac * canvasRect.width}px`;
    marker.style.top = `${canvasRect.top - containerRect.top + yFrac * canvasRect.height}px`;
    container.appendChild(marker);
}

function clearPointMarkers() {
    ui.mainCanvas.parentElement?.querySelectorAll(".mask-point-marker")
        .forEach(el => el.remove());
}

/**
 * 現在の編集（全マスク込み）を適用したフル解像度画像から RawImage を作成する（SAM2への入力用）。
 */
async function captureFullResRawImage(): Promise<InstanceType<typeof RawImage>> {
    if (!editorFull) {
        throw new Error("No image loaded.");
    }

    applyAllMasksToEditor(editorFull);
    editorFull.apply();

    const rgb = await editorFull.get_rgb_f32() as Float32Array;
    const width = editorFull.width();
    const height = editorFull.height();

    const uint8 = new Uint8ClampedArray(rgb.length);
    for (let i = 0; i < rgb.length; i++) {
        uint8[i] = Math.max(0, Math.min(255, Math.round(rgb[i] * 255)));
    }

    return new RawImage(uint8, width, height, 3);
}

/**
 * 新規マスクをキャッシュ・UI・各解像度のエディタすべてに登録し、選択状態にする。
 * data は二値化前の生データ（SAM2ロジット等）。
 */
function registerNewMask(data: Float32Array, width: number, height: number): string {
    const name = `${maskCounter++}`;
    const params = createDefaultParameters();

    maskDataCache.set(name, { data, width, height });
    maskOverlayCache.delete(name);

    masks.push({
        name,
        edit_parameters: params,
        data,
    });

    addMaskToAllEditors(name, data, width, height);

    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    ui.maskSelect.appendChild(option);

    ui.maskSelect.value = name;
    switchToMaskEditState(name);
    return name;
}

/**
 * UIの編集対象を指定マスクに切り替える。トーンカーブUI・スライダー・有効/無効状態を全て同期する。
 */
function switchToMaskEditState(name: string) {
    const entry = getMaskEntry(name);
    if (!entry) return;

    selectedMaskName = name;
    editState = entry.edit_parameters;

    updateAllSliderLabels();
    syncToneCurveEditorsFromState();
    updateMaskControlsEnabled();
    updateImage();
}

function syncToneCurveEditorsFromState() {
    toneCurveEditors['brightness'].points = editState.brightness_tone_curve_points;
    toneCurveEditors['hue'].points = editState.hue_tone_curve_points;
    toneCurveEditors['saturation'].points = editState.saturation_tone_curve_points;
    toneCurveEditors['lightness'].points = editState.lightness_tone_curve_points;

    Object.values(toneCurveEditors).forEach(editor => editor.draw());
}

function updateMaskControlsEnabled() {
    const isMain = selectedMaskName === "main";
    ui.btnDeleteMask.disabled = isMain;
    ui.btnInvertMask.disabled = isMain;
    ui.maskRangeSlider.disabled = isMain;
}

function addMaskToAllEditors(name: string, data: Float32Array, width: number, height: number) {
    const src: Float32Mask = { data, width, height };

    if (editorFull) {
        const r = resizeFloat32SingleChannelIfNeeded(src, editorFull.width(), editorFull.height());
        editorFull.add_mask(name, r.data, r.width, r.height);
    }
    if (editorMid) {
        const r = resizeFloat32SingleChannelIfNeeded(src, editorMid.width(), editorMid.height());
        editorMid.add_mask(name, r.data, r.width, r.height);
    }
    if (editorLow) {
        const r = resizeFloat32SingleChannelIfNeeded(src, editorLow.width(), editorLow.height());
        editorLow.add_mask(name, r.data, r.width, r.height);
    }
}

function removeMaskFromAllEditors(name: string) {
    editorFull?.remove_mask(name);
    editorMid?.remove_mask(name);
    editorLow?.remove_mask(name);
}

/**
 * 表示用にマスク（生データ）を指定解像度へリサイズする（キャッシュ付き）。
 */
function getMaskForDisplay(name: string, width: number, height: number): Float32Mask | undefined {
    const canonical = maskDataCache.get(name);
    if (!canonical) return undefined;

    if (canonical.width === width && canonical.height === height) {
        return canonical;
    }

    const cached = maskOverlayCache.get(name);
    if (cached && cached.width === width && cached.height === height) {
        return cached;
    }

    const resized = resizeFloat32SingleChannel(canonical, width, height);
    maskOverlayCache.set(name, resized);
    return resized;
}

/**
 * 選択中マスクを、表示専用に「そのマスクのmask_rangeで二値化」して赤半透明でrgbaにオーバーレイする。
 * ここでの二値化は表示だけのためのもので、add_maskに渡すデータには一切影響しない。
 */
function applyMaskOverlayIfNeeded(rgba: Float32Array, width: number, height: number) {
    if (!showMask || selectedMaskName === "main") return;

    const mask = getMaskForDisplay(selectedMaskName, width, height);
    if (!mask) return;

    const threshold = editState.mask_range;
    const data = mask.data;
    const pixelCount = width * height;
    const alpha = 0.5;

    for (let i = 0; i < pixelCount; i++) {
        if (data[i] <= threshold) continue;

        const idx = i * 4;
        rgba[idx] = rgba[idx] * (1 - alpha) + 1.0 * alpha;     // R
        rgba[idx + 1] = rgba[idx + 1] * (1 - alpha);           // G
        rgba[idx + 2] = rgba[idx + 2] * (1 - alpha);           // B
    }
}

// ============================================================

function setupToneCurveEditors() {
    const onCurveChange = (key: keyof EditState) => (points: { x: number, y: number }[]) => {
        (editState as any)[key] = points;
        updateImage();
    };

    toneCurveEditors['brightness'] = new ToneCurveEditor('brightness-tone-curve-editor', CurveMode.BRIGHTNESS, onCurveChange('brightness_tone_curve_points'), onDragStart, onDragEnd);
    toneCurveEditors['hue'] = new ToneCurveEditor('hue-tone-curve-editor', CurveMode.HUE, onCurveChange('hue_tone_curve_points'), onDragStart, onDragEnd);
    toneCurveEditors['saturation'] = new ToneCurveEditor('saturation-tone-curve-editor', CurveMode.SATURATION, onCurveChange('saturation_tone_curve_points'), onDragStart, onDragEnd);
    toneCurveEditors['lightness'] = new ToneCurveEditor('lightness-tone-curve-editor', CurveMode.LIGHTNESS, onCurveChange('lightness_tone_curve_points'), onDragStart, onDragEnd);

    toneCurveEditors['brightness'].setBackground('./assets/tone_curve/brightness_gradient.png');
    toneCurveEditors['hue'].setBackground('./assets/tone_curve/hue_bars.png');
    toneCurveEditors['saturation'].setBackground('./assets/tone_curve/hue_vs_saturation.png');
    toneCurveEditors['lightness'].setBackground('./assets/tone_curve/hue_vs_lightness.png');

    syncToneCurveEditorsFromState();
}

function updateMetadataTableFromJson(json: string): void {
    const tbody = document.querySelector("#metadata-table tbody") as HTMLTableSectionElement;
    tbody.innerHTML = "";

    let metadata: Record<string, unknown>;

    try {
        metadata = JSON.parse(json);
    } catch {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 2;
        td.textContent = "Failed to parse JSON.";
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
    }

    for (const [key, value] of Object.entries(metadata)) {
        const tr = document.createElement("tr");
        const keyTd = document.createElement("td");
        keyTd.textContent = key;
        const valueTd = document.createElement("td");
        valueTd.textContent = String(value);
        tr.appendChild(keyTd);
        tr.appendChild(valueTd);
        tbody.appendChild(tr);
    }
}

async function loadImage(file: File) {
    const midResLongEdge = settings.uiPreviewSize;
    const lowResLongEdge = settings.dragPreviewSize;

    // 新しい画像を読み込むので、既存のマスクはすべて破棄する
    resetMasks();

    editorFull = new PhotoEditor(gpuProcessor, await file.bytes(), file.name.split('.').pop()?.toLowerCase() as string);

    const rgb = await editorFull.get_rgb_f32();

    const float32ArrayImageFull: Float32ArrayImageRGB = {
        data: rgb as Float32Array<ArrayBuffer>,
        width: editorFull.width(),
        height: editorFull.height()
    };

    const float32ArrayImageMid = resizeFloat32RGBLongEdge(float32ArrayImageFull, midResLongEdge);
    editorMid = PhotoEditor.create_from_rgb_f32(gpuProcessor, float32ArrayImageMid.data, float32ArrayImageMid.width, float32ArrayImageMid.height);

    const float32ArrayImageLow = resizeFloat32RGBLongEdge(float32ArrayImageFull, lowResLongEdge);
    editorLow = PhotoEditor.create_from_rgb_f32(gpuProcessor, float32ArrayImageLow.data, float32ArrayImageLow.width, float32ArrayImageLow.height);

    ui.mainCanvas.width = editorFull.width();
    ui.mainCanvas.height = editorFull.height();

    imageLoaded = true;

    updateMetadataTableFromJson(editorFull.exif_json());
    resetAllEdits();
}

function resetMasks() {
    exitMaskCreationMode();
    maskDataCache.clear();
    maskOverlayCache.clear();

    masks = [{ name: "main", edit_parameters: createDefaultParameters(), data: null }];
    maskCounter = 1;
    showMask = false;

    mainEditParams = masks[0].edit_parameters;
    editState = masks[0].edit_parameters;
    selectedMaskName = "main";

    ui.maskSelect.innerHTML = "";
    const mainOption = document.createElement("option");
    mainOption.value = "main";
    mainOption.textContent = "main";
    ui.maskSelect.appendChild(mainOption);
    ui.maskSelect.value = "main";
    ui.chkShowMask.checked = false;

    updateMaskControlsEnabled();
}

async function renderProcessedTextureToCanvas(
    textureView: GPUTextureView,
    width: number,
    height: number
) {
    if (!canvasContext || !renderPipeline) return;

    const device = gpu.device;
    const queue = gpu.queue;

    const data = new Float32Array([
        ui.mainCanvas.width,
        ui.mainCanvas.height,
        width,
        height
    ]);
    queue.writeBuffer(uniformBuffer, 0, data.buffer);

    const bindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: textureView },
            { binding: 1, resource: { buffer: uniformBuffer } },
        ],
    });

    const canvasTexture = canvasContext.getCurrentTexture();

    const commandEncoder = device.createCommandEncoder();

    const pass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: canvasTexture.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
        }],
    });

    pass.setPipeline(renderPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(4);
    pass.end();

    queue.submit([commandEncoder.finish()]);
}

async function updateImage() {
    if (isRendering) {
        return;
    }
    isRendering = true;

    if (!imageLoaded || !editorFull || !editorMid || !editorLow) {
        isRendering = false;
        return;
    }

    let editor: PhotoEditor;
    switch (previewLevel) {
        case PreviewLevel.LOW: editor = editorLow; break;
        case PreviewLevel.MID: editor = editorMid; break;
        case PreviewLevel.FULL: editor = editorFull; break;
    }

    applyAllMasksToEditor(editor);
    editor.apply();

    const rgba = await editor.get_rgba_f32() as Float32Array;
    const width = editor.width();
    const height = editor.height();

    applyMaskOverlayIfNeeded(rgba, width, height);

    const textureView = await uploadFloat32Texture(rgba, width, height);
    await renderProcessedTextureToCanvas(textureView, width, height);

    isRendering = false;
}

/**
 * main + 全マスクぶんのトーン/WB/カーブ/mask_rangeを指定エディタに適用する。
 * vignette / lens_distortion はマスク非対応のため常にmainの値を使う。
 */
function applyAllMasksToEditor(e: PhotoEditor) {
    e.set_vignette(mainEditParams.vignette);
    e.set_lens_distortion_correction(mainEditParams.lens_distortion);

    const toPoints = (points: { x: number, y: number }[]) => points.map(p => p.x * 65535);
    const toValues = (points: { x: number, y: number }[]) => points.map(p => p.y * 65535);
    const toSatLightValues = (points: { x: number, y: number }[]) => points.map(p => p.y / 2 * 65535);

    for (const maskEntry of masks) {
        const s = maskEntry.edit_parameters;
        const isMain = maskEntry.name === "main";
        const maskNameArg = isMain ? undefined : maskEntry.name;

        e.set_tone(s.exposure, s.contrast, s.shadow, s.highlight, s.black, s.white, maskNameArg);
        e.set_whitebalance(s.temperature, s.tint, maskNameArg);

        e.set_brightness_tone_curve(
            undefined,
            new Int32Array(toPoints(s.brightness_tone_curve_points)),
            new Int32Array(toValues(s.brightness_tone_curve_points)),
            maskNameArg
        );
        e.set_oklch_hue_curve(
            undefined,
            new Int32Array(toPoints(s.hue_tone_curve_points)),
            new Int32Array(toValues(s.hue_tone_curve_points)),
            maskNameArg
        );
        e.set_oklch_saturation_curve(
            undefined,
            new Int32Array(toPoints(s.saturation_tone_curve_points)),
            new Int32Array(toSatLightValues(s.saturation_tone_curve_points)),
            maskNameArg
        );
        e.set_oklch_lightness_curve(
            undefined,
            new Int32Array(toPoints(s.lightness_tone_curve_points)),
            new Int32Array(toSatLightValues(s.lightness_tone_curve_points)),
            maskNameArg
        );

        if (!isMain) {
            e.set_mask_range(maskEntry.name, s.mask_range);
        }
    }
}

function resetAllEdits() {
    resetCurve('brightness');
    resetCurve('hue');
    resetCurve('saturation');
    resetCurve('lightness');
    resetTone();
    resetWb();
    resetEffect();
}

function resetTone() {
    editState.exposure = 0.0;
    editState.contrast = 0;
    editState.shadow = 0;
    editState.highlight = 0;
    editState.black = 0;
    editState.white = 0;
    updateAllSliderLabels();
    updateImage();
}
function resetWb() {
    editState.temperature = 0;
    editState.tint = 0;
    updateAllSliderLabels();
    updateImage();
}
function resetEffect() {
    mainEditParams.vignette = 0;
    mainEditParams.lens_distortion = 0;
    updateAllSliderLabels();
    updateImage();
}
function resetCurve(name: string) {
    if (toneCurveEditors[name]) {
        toneCurveEditors[name].initializePoints();
        if (name === "brightness" || name === "hue") {
            toneCurveEditors[name].points = [{ x: 0.0, y: 0.0 }, { x: 1.0, y: 1.0 }];
        } else {
            toneCurveEditors[name].points = [{ x: 0.0, y: 1.0 }, { x: 1.0, y: 1.0 }];
        }

        (editState as any)[`${name}_tone_curve_points`] = toneCurveEditors[name].points;
        updateImage();
    }
}

function updateMaskRangeLabel() {
    ui.maskRangeSlider.value = editState.mask_range.toString();
    ui.maskRangeLabel.textContent = `${i18n.t("TR_MASK_RANGE")}: ${editState.mask_range.toFixed(1)}`;
}

function updateAllSliderLabels() {
    ui.exposureSlider.value = editState.exposure.toString();
    ui.exposureLabel.textContent = `${i18n.t("TR_EXPOSURE")} ${editState.exposure.toFixed(2)}`;
    ui.contrastSlider.value = editState.contrast.toString();
    ui.contrastLabel.textContent = `${i18n.t("TR_CONTRAST")} ${editState.contrast}`;
    ui.shadowSlider.value = editState.shadow.toString();
    ui.shadowLabel.textContent = `${i18n.t("TR_SHADOW")} ${editState.shadow}`;
    ui.highlightSlider.value = editState.highlight.toString();
    ui.highlightLabel.textContent = `${i18n.t("TR_HIGHLIGHT")} ${editState.highlight}`;
    ui.blackSlider.value = editState.black.toString();
    ui.blackLabel.textContent = `${i18n.t("TR_BLACK_LEVEL")} ${editState.black}`;
    ui.whiteSlider.value = editState.white.toString();
    ui.whiteLabel.textContent = `${i18n.t("TR_WHITE_LEVEL")} ${editState.white}`;
    ui.temperatureSlider.value = editState.temperature.toString();
    ui.temperatureLabel.textContent = `${i18n.t("TR_TEMPERATURE")} ${editState.temperature}`;
    ui.tintSlider.value = editState.tint.toString();
    ui.tintLabel.textContent = `${i18n.t("TR_TINT")} ${editState.tint}`;

    // vignette / lens_distortion は常にmain固定
    ui.vignetteSlider.value = mainEditParams.vignette.toString();
    ui.vignetteLabel.textContent = `${i18n.t("TR_VIGNETTE")} ${mainEditParams.vignette}`;
    ui.lensDistortionSlider.value = mainEditParams.lens_distortion.toString();
    ui.lensDistortionLabel.textContent = `${i18n.t("TR_LENS_DISTORTION")} ${mainEditParams.lens_distortion}`;

    updateMaskRangeLabel();
}

function onDragStart() {
    previewLevel = PreviewLevel.LOW;
    updateImage();
}
function onDragEnd() {
    previewLevel = PreviewLevel.MID;
    updateImage();
}

async function saveImage() {
    if (!editorFull || !currentImageFile) return;

    ui.saveDialog.style.display = 'none';

    previewLevel = PreviewLevel.FULL;
    applyAllMasksToEditor(editorFull);
    editorFull.apply();

    let bytes: Uint8Array<ArrayBuffer>;

    if (ui.formatSelect.value === "jpeg") {
        bytes = await editorFull.save_jpeg() as Uint8Array<ArrayBuffer>;
    } else {
        bytes = await editorFull.save_png() as Uint8Array<ArrayBuffer>;
    }

    const blob = new Blob(
        [bytes],
        {
            type:
                ui.formatSelect.value === "jpeg"
                    ? "image/jpeg"
                    : "image/png"
        }
    );

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const basename = currentImageFile.name.split('.').slice(0, -1).join('.');
    const ext = ui.formatSelect.value;
    a.href = url;
    a.download = `${basename}_edited.${ext}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function resizeFloat32RGBLongEdge(
    src: Float32ArrayImageRGB,
    targetLongEdge: number
): Float32ArrayImageRGB {

    const srcWidth = src.width;
    const srcHeight = src.height;
    const srcData = src.data;

    let dstWidth: number;
    let dstHeight: number;

    if (srcWidth >= srcHeight) {
        dstWidth = targetLongEdge;
        dstHeight = Math.round(srcHeight * (targetLongEdge / srcWidth));
    } else {
        dstHeight = targetLongEdge;
        dstWidth = Math.round(srcWidth * (targetLongEdge / srcHeight));
    }

    const dst = new Float32Array(dstWidth * dstHeight * 3);

    const scaleX = srcWidth / dstWidth;
    const scaleY = srcHeight / dstHeight;

    for (let y = 0; y < dstHeight; y++) {
        const sy = (y + 0.5) * scaleY - 0.5;
        const y0 = Math.max(Math.floor(sy), 0);
        const y1 = Math.min(y0 + 1, srcHeight - 1);
        const ty = sy - y0;

        for (let x = 0; x < dstWidth; x++) {
            const sx = (x + 0.5) * scaleX - 0.5;
            const x0 = Math.max(Math.floor(sx), 0);
            const x1 = Math.min(x0 + 1, srcWidth - 1);
            const tx = sx - x0;

            const i00 = (y0 * srcWidth + x0) * 3;
            const i10 = (y0 * srcWidth + x1) * 3;
            const i01 = (y1 * srcWidth + x0) * 3;
            const i11 = (y1 * srcWidth + x1) * 3;

            const di = (y * dstWidth + x) * 3;

            for (let c = 0; c < 3; c++) {
                const c00 = srcData[i00 + c];
                const c10 = srcData[i10 + c];
                const c01 = srcData[i01 + c];
                const c11 = srcData[i11 + c];

                const cx0 = c00 * (1.0 - tx) + c10 * tx;
                const cx1 = c01 * (1.0 - tx) + c11 * tx;

                dst[di + c] = cx0 * (1.0 - ty) + cx1 * ty;
            }
        }
    }

    return { data: dst, width: dstWidth, height: dstHeight };
}

/**
 * 単一チャンネル（マスク生データ）を指定の幅・高さへバイリニアでリサイズする。
 */
function resizeFloat32SingleChannel(src: Float32Mask, dstWidth: number, dstHeight: number): Float32Mask {
    const { width: srcWidth, height: srcHeight, data: srcData } = src;

    const dst = new Float32Array(dstWidth * dstHeight);
    const scaleX = srcWidth / dstWidth;
    const scaleY = srcHeight / dstHeight;

    for (let y = 0; y < dstHeight; y++) {
        const sy = (y + 0.5) * scaleY - 0.5;
        const y0 = Math.max(Math.floor(sy), 0);
        const y1 = Math.min(y0 + 1, srcHeight - 1);
        const ty = sy - y0;

        for (let x = 0; x < dstWidth; x++) {
            const sx = (x + 0.5) * scaleX - 0.5;
            const x0 = Math.max(Math.floor(sx), 0);
            const x1 = Math.min(x0 + 1, srcWidth - 1);
            const tx = sx - x0;

            const c00 = srcData[y0 * srcWidth + x0];
            const c10 = srcData[y0 * srcWidth + x1];
            const c01 = srcData[y1 * srcWidth + x0];
            const c11 = srcData[y1 * srcWidth + x1];

            const cx0 = c00 * (1.0 - tx) + c10 * tx;
            const cx1 = c01 * (1.0 - tx) + c11 * tx;

            dst[y * dstWidth + x] = cx0 * (1.0 - ty) + cx1 * ty;
        }
    }

    return { data: dst, width: dstWidth, height: dstHeight };
}

function resizeFloat32SingleChannelIfNeeded(src: Float32Mask, dstWidth: number, dstHeight: number): Float32Mask {
    if (src.width === dstWidth && src.height === dstHeight) {
        return { data: src.data.slice(), width: dstWidth, height: dstHeight };
    }
    return resizeFloat32SingleChannel(src, dstWidth, dstHeight);
}

initializeApp();