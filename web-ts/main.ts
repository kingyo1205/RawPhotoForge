
import { GpuImageProcessor } from './core/gpu_image_processor.ts';
import { PhotoEditor } from './core/photo_editor.ts';
import { loadPpm, Float32ArrayImage, resizeFloat32RGBALongEdge, imageBitmapToFloat32RGBA } from './core/image.ts';
import { ToneCurveEditor, CurveMode, Point } from './tone_curve_editor.ts';
import translation from "./translations/translation.json" with { type: "text" };

type Translations = Record<string, Record<string, string>>;


interface Settings {
    uiPreviewSize: number;
    dragPreviewSize: number;
    locale: string;
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
        console.log(data);
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

type EditState = {
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
};

enum PreviewLevel { LOW, MID, FULL }


let gpuProcessor: GpuImageProcessor;
let editorFull: PhotoEditor | null = null;
let editorMid: PhotoEditor | null = null;
let editorLow: PhotoEditor | null = null;

let currentImageFile: File | null = null;
let imageLoaded = false;
let previewLevel = PreviewLevel.MID;

let uniformBuffer: GPUBuffer;

let toneCurveEditors: { [key: string]: ToneCurveEditor } = {};

const initialEditState: EditState = {
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
};

let editState: EditState = { ...initialEditState };


let canvasContext: GPUCanvasContext | null = null;
let presentationFormat: GPUTextureFormat;
let renderPipeline: GPURenderPipeline | null = null;
let isRendering = false;

const data: Translations = JSON.parse(translation);
console.log(data);
const i18n: I18n = new I18n(data, "en");

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

    metadataTree: document.getElementById('metadata-tree') as HTMLDivElement,
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
    if (i18n.setLang) {
        i18n.setLang(settings.locale);
    }
    applyI18n(i18n);
    updateAllSliderLabels();
    console.log("Applied settings:", settings);
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

    setupEventListeners();
    setupToneCurveEditors();
    updateAllSliderLabels();


    canvasContext = ui.mainCanvas.getContext("webgpu");
    if (!canvasContext) {
        alert(i18n.t("TR_ERROR_GET_WEBGPU_CANVAS_CONTEXT"));
        return;
    }
    presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    canvasContext.configure({
        device: gpuProcessor.getDevice(),
        format: presentationFormat,
        alphaMode: 'premultiplied',
    });

    const device = gpuProcessor.getDevice();
    const shaderModule = device.createShaderModule({
        label: "Render Shader Module",
        code: renderShaderCode,
    });

    const renderBindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                texture: {
                    sampleType: 'unfilterable-float',
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: {
                    type: 'uniform',
                },
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


    const sliders = [
        { s: ui.exposureSlider, k: 'exposure', l: ui.exposureLabel, n: i18n.t("TR_EXPOSURE"), f: (v: number) => v.toFixed(2) },
        { s: ui.contrastSlider, k: 'contrast', l: ui.contrastLabel, n: i18n.t("TR_CONTRAST"), f: (v: number) => Math.round(v) },
        { s: ui.shadowSlider, k: 'shadow', l: ui.shadowLabel, n: i18n.t("TR_SHADOW"), f: (v: number) => Math.round(v) },
        { s: ui.highlightSlider, k: 'highlight', l: ui.highlightLabel, n: i18n.t("TR_HIGHLIGHT"), f: (v: number) => Math.round(v) },
        { s: ui.blackSlider, k: 'black', l: ui.blackLabel, n: i18n.t("TR_BLACK_LEVEL"), f: (v: number) => Math.round(v) },
        { s: ui.whiteSlider, k: 'white', l: ui.whiteLabel, n: i18n.t("TR_WHITE_LEVEL"), f: (v: number) => Math.round(v) },
        { s: ui.temperatureSlider, k: 'temperature', l: ui.temperatureLabel, n: i18n.t("TR_TEMPERATURE"), f: (v: number) => Math.round(v) },
        { s: ui.tintSlider, k: 'tint', l: ui.tintLabel, n: i18n.t("TR_TINT"), f: (v: number) => Math.round(v) },
        { s: ui.vignetteSlider, k: 'vignette', l: ui.vignetteLabel, n: i18n.t("TR_VIGNETTE"), f: (v: number) => Math.round(v) },
        { s: ui.lensDistortionSlider, k: 'lens_distortion', l: ui.lensDistortionLabel, n: i18n.t("TR_LENS_DISTORTION"), f: (v: number) => Math.round(v) },
    ];

    sliders.forEach(({ s, k, l, n, f }) => {
        s.addEventListener('input', () => {
            const value = parseFloat(s.value);
            (editState as any)[k] = value;
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

    Object.values(toneCurveEditors).forEach(editor => {
        editState.brightness_tone_curve_points = editor.points;
    });
}


async function loadImage(file: File) {
    let originalImageBitmap: ImageBitmap;
    let float32ArrayImage;

    const midResLongEdge = settings.uiPreviewSize;
    const lowResLongEdge = settings.dragPreviewSize;

    if (file.name.toLowerCase().endsWith(".ppm")) {
        float32ArrayImage = await loadPpm(file)


    } else {
        originalImageBitmap = await createImageBitmap(file);
        float32ArrayImage = await imageBitmapToFloat32RGBA(originalImageBitmap);

    }

    editorFull = await PhotoEditor.create(gpuProcessor, float32ArrayImage);

    editorMid = await PhotoEditor.create(gpuProcessor, resizeFloat32RGBALongEdge(float32ArrayImage, midResLongEdge));

    editorLow = await PhotoEditor.create(gpuProcessor, resizeFloat32RGBALongEdge(float32ArrayImage, lowResLongEdge));


    ui.mainCanvas.width = editorFull.image.width;
    ui.mainCanvas.height = editorFull.image.height;


    imageLoaded = true;
    resetAllEdits();

}

async function resizeBitmap(bitmap: ImageBitmap, longEdge: number): Promise<ImageBitmap> {
    const scale = longEdge / Math.max(bitmap.width, bitmap.height);
    if (scale >= 1) return bitmap;

    const newWidth = Math.round(bitmap.width * scale);
    const newHeight = Math.round(bitmap.height * scale);

    return await createImageBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, {
        resizeWidth: newWidth,
        resizeHeight: newHeight,
        resizeQuality: 'high'
    });
}

async function renderProcessedTextureToCanvas(
    textureView: GPUTextureView,
    width: number,
    height: number
) {
    if (!canvasContext || !renderPipeline) return;

    const device = gpuProcessor.getDevice();
    const queue = gpuProcessor.getQueue();


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
    let s = Date.now();
    let editor: PhotoEditor;
    switch (previewLevel) {
        case PreviewLevel.LOW: editor = editorLow; break;
        case PreviewLevel.MID: editor = editorMid; break;
        case PreviewLevel.FULL: editor = editorFull; break;
    }

    setEditorParameters(editor);

    await editor.applyAdjustments();
    console.log("applyAdjustments", Date.now() - s);





    await renderProcessedTextureToCanvas(editor.image.textureView, editor.image.width, editor.image.height);



    isRendering = false;

    console.log("applyAdjustments+render", Date.now() - s);
}

function setEditorParameters(e: PhotoEditor) {
    const s = editState;

    e.setTone(s.exposure, s.contrast, s.shadow, s.highlight, s.black, s.white);
    e.setWhitebalance(s.temperature, s.tint);
    e.setVignette(s.vignette);
    e.setLensDistortionCorrection(s.lens_distortion);

    const toPoints = (points: { x: number, y: number }[]) => points.map(p => p.x * 65535);
    const toValues = (points: { x: number, y: number }[]) => points.map(p => p.y * 65535);

    e.setBrightnessToneCurve(undefined, toPoints(s.brightness_tone_curve_points), toValues(s.brightness_tone_curve_points));
    e.setOklchHueCurve(undefined, toPoints(s.hue_tone_curve_points), toValues(s.hue_tone_curve_points));

    const toSatLightValues = (points: { x: number, y: number }[]) => points.map(p => p.y / 2 * 65535);
    e.setOklchSaturationCurve(undefined, toPoints(s.saturation_tone_curve_points), toSatLightValues(s.saturation_tone_curve_points));
    e.setOklchLightnessCurve(undefined, toPoints(s.lightness_tone_curve_points), toSatLightValues(s.lightness_tone_curve_points));
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
    editState.vignette = 0;
    editState.lens_distortion = 0;
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
    ui.vignetteSlider.value = editState.vignette.toString();
    ui.vignetteLabel.textContent = `${i18n.t("TR_VIGNETTE")} ${editState.vignette}`;
    ui.lensDistortionSlider.value = editState.lens_distortion.toString();
    ui.lensDistortionLabel.textContent = `${i18n.t("TR_LENS_DISTORTION")} ${editState.lens_distortion}`;
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
    setEditorParameters(editorFull);
    await editorFull.applyAdjustments();
    const finalBitmap = await editorFull.save();


    const canvas = document.createElement('canvas');
    canvas.width = finalBitmap.width;
    canvas.height = finalBitmap.height;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(finalBitmap, 0, 0);

    const format = ui.formatSelect.value === 'jpeg' ? 'image/jpeg' : 'image/png';
    const quality = format === 'image/jpeg' ? 0.9 : undefined;
    const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, format, quality));

    if (!blob) {
        alert(i18n.t("TR_ERROR_IMAGE_SAVE"));
        return;
    }


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



initializeApp();
