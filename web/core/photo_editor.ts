



import { GpuImageProcessor } from "./gpu_image_processor.ts";
import { Float32ArrayImage, Image } from "./image.ts";
import { pchipInterpolate } from "./interpolation.ts";

export const CURVE_RESOLUTION: number = 65536;

export interface EditParameters {

    exposure: number;
    contrast: number;
    shadow: number;
    highlight: number;
    black: number;
    white: number;

    wb_temperature: number;
    wb_tint: number;

    vignette: number;

    lens_distortion: number;

    brightness_tone_curve: Int32Array;
    hue_tone_curve: Int32Array;
    saturation_tone_curve: Int32Array;
    lightness_tone_curve: Int32Array;
}

export interface Mask {
    name: string;
    gpuMask: GpuMask;
    editParameters: EditParameters;
}

export interface GpuMask {
    texture: GPUTexture;
    view: GPUTextureView;
}

export class PhotoEditor {
    private gpuProcessor: GpuImageProcessor;
    public image: Image;
    public originalImage: Image;
    public masks: Mask[] = [];

    private constructor(gpuProcessor: GpuImageProcessor, image: Image) {
        this.gpuProcessor = gpuProcessor;
        this.image = image;
        this.originalImage = image.clone();

        const mainMask = this.createGpuMask(image.width, image.height);
        this.masks.push({
            name: "main",
            gpuMask: mainMask,
            editParameters: createDefaultEditParameters(),
        });
    }

    public static async create(
        gpuProcessor: GpuImageProcessor,
        imageData: Float32ArrayImage,
    ): Promise<PhotoEditor> {

        const image = await Image.createFromFloatData(gpuProcessor, imageData);
        return new PhotoEditor(gpuProcessor, image);
    }

    public createGpuMask(
        width: number,
        height: number,
        data?: Float32Array<ArrayBuffer>,
    ): GpuMask {
        const device = this.gpuProcessor.getDevice();
        const texture = device.createTexture({
            label: "Mask Texture",
            size: { width, height },
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
        });

        const maskData = data ?? new Float32Array(width * height).fill(1.0);

        this.gpuProcessor.getQueue().writeTexture(
            { texture },
            maskData.buffer,
            { bytesPerRow: width * 4 },
            { width, height }
        );

        return { texture, view: texture.createView() };
    }

    public async save(): Promise<ImageBitmap> {
        const rgbaPixels = await this.image.readRgbaPixels();


        const clampedPixels = new Uint8ClampedArray(rgbaPixels.length);
        for (let i = 0; i < rgbaPixels.length; i++) {
            clampedPixels[i] = rgbaPixels[i] * 255;
        }

        const imageData = new ImageData(clampedPixels, this.image.width, this.image.height);
        return await createImageBitmap(imageData);
    }

    public reset() {
        this.image = this.originalImage.clone();
        this.masks = this.masks.filter(m => m.name === "main");
        const mainMask = this.masks.find(m => m.name === "main");
        if (mainMask) {
            mainMask.editParameters = createDefaultEditParameters();
        }
    }

    private getAdjustmentSet(maskName?: string): EditParameters {
        const name = maskName ?? "main";
        const mask = this.masks.find(m => m.name === name);
        if (!mask) {
            throw new Error(`The specified mask '${name}' does not exist.`);
        }
        return mask.editParameters;
    }

    public setWhitebalance(temperature: number, tint: number, maskName?: string) {
        const adjustments = this.getAdjustmentSet(maskName);
        adjustments.wb_temperature = clamp(temperature, -100, 100);
        adjustments.wb_tint = clamp(tint, -100, 100);
    }

    public setVignette(value: number) {
        this.getAdjustmentSet("main").vignette = clamp(value, -100, 100);
    }

    public setLensDistortionCorrection(value: number) {
        this.getAdjustmentSet("main").lens_distortion = clamp(value, -100, 100);
    }

    public setTone(
        exposure: number, contrast: number, shadow: number,
        highlight: number, black: number, white: number, maskName?: string
    ) {
        const adjustments = this.getAdjustmentSet(maskName);
        adjustments.exposure = clamp(exposure, -10.0, 10.0);
        adjustments.contrast = clamp(contrast, -100, 100);
        adjustments.shadow = clamp(shadow, -100, 100);
        adjustments.highlight = clamp(highlight, -100, 100);
        adjustments.black = clamp(black, -100, 100);
        adjustments.white = clamp(white, -100, 100);
    }

    private _setCurve(
        curveType: keyof Pick<EditParameters, 'brightness_tone_curve' | 'hue_tone_curve' | 'saturation_tone_curve' | 'lightness_tone_curve'>,
        curve: Int32Array | undefined,
        controlPointsX: number[] | undefined,
        controlPointsY: number[] | undefined,
        maskName: string | undefined
    ) {
        if (curve === undefined && controlPointsX === undefined) {
            throw new Error("You must provide either a full curve or control points.");
        }

        let finalCurve: Int32Array;
        if (curve) {
            if (curve.length !== CURVE_RESOLUTION) {
                throw new Error(`Invalid curve length. Expected ${CURVE_RESOLUTION}, but got ${curve.length}.`);
            }
            finalCurve = curve;
        } else {
            if (!controlPointsX || !controlPointsY) {
                throw new Error("Control points for x and y must be provided together.");
            }
            if (controlPointsX.length !== controlPointsY.length) {
                throw new Error("Control point arrays must have the same length.");
            }
            if (controlPointsX.length === 0) {
                throw new Error("Control point arrays cannot be empty.");
            }

            const xEval = new Float32Array(CURVE_RESOLUTION).map((_, i) => i);
            const interpolated = pchipInterpolate(controlPointsX, controlPointsY, xEval);
            finalCurve = new Int32Array(interpolated.map(v => clamp(v, 0, 65535)));
        }
        this.getAdjustmentSet(maskName)[curveType] = finalCurve;
    }

    public setBrightnessToneCurve(curve: Int32Array | undefined, controlPointsX: number[] | undefined, controlPointsY: number[] | undefined, maskName?: string) {
        this._setCurve('brightness_tone_curve', curve, controlPointsX, controlPointsY, maskName);
    }
    public setOklchHueCurve(curve: Int32Array | undefined, controlPointsX: number[] | undefined, controlPointsY: number[] | undefined, maskName?: string) {
        this._setCurve('hue_tone_curve', curve, controlPointsX, controlPointsY, maskName);
    }
    public setOklchSaturationCurve(curve: Int32Array | undefined, controlPointsX: number[] | undefined, controlPointsY: number[] | undefined, maskName?: string) {
        this._setCurve('saturation_tone_curve', curve, controlPointsX, controlPointsY, maskName);
    }
    public setOklchLightnessCurve(curve: Int32Array | undefined, controlPointsX: number[] | undefined, controlPointsY: number[] | undefined, maskName?: string) {
        this._setCurve('lightness_tone_curve', curve, controlPointsX, controlPointsY, maskName);
    }

    public addMask(name: string, maskData: Float32Array<ArrayBuffer>) {
        const gpuMask = this.createGpuMask(this.originalImage.width, this.originalImage.height, maskData);
        this.masks.push({ name, gpuMask, editParameters: createDefaultEditParameters() });
    }

    public removeMask(name: string) {
        if (name === "main") {
            throw new Error("The main mask cannot be deleted.");
        }
        this.masks = this.masks.filter(mask => mask.name !== name);
    }

    public async applyAdjustments() {
        this.image = await this.gpuProcessor.applyAdjustments(this.originalImage, this.masks);
    }
}

function clamp(num: number, min: number, max: number): number {
    return Math.min(Math.max(num, min), max);
}

function createDefaultEditParameters(): EditParameters {
    const resolutionRange = new Int32Array(CURVE_RESOLUTION).map((_, i) => i);
    return {
        exposure: 0.0,
        contrast: 0,
        shadow: 0,
        highlight: 0,
        black: 0,
        white: 0,
        wb_temperature: 0,
        wb_tint: 0,
        vignette: 0,
        lens_distortion: 0,
        brightness_tone_curve: resolutionRange,
        hue_tone_curve: resolutionRange,
        saturation_tone_curve: new Int32Array(CURVE_RESOLUTION).fill(32767),
        lightness_tone_curve: new Int32Array(CURVE_RESOLUTION).fill(32767),
    };
}

