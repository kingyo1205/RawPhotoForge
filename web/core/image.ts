

import type { GpuImageProcessor } from "./gpu_image_processor.ts";

export interface Float32ArrayImage {
    data: Float32Array<ArrayBuffer>,
    width: number,
    height: number
}

export class Image {
    private gpuProcessor: GpuImageProcessor;
    public readonly texture: GPUTexture;
    public readonly textureView: GPUTextureView;
    public readonly width: number;
    public readonly height: number;

    private destroyed = false;

    constructor(
        gpuProcessor: GpuImageProcessor,
        texture: GPUTexture,
        width: number,
        height: number
    ) {
        this.gpuProcessor = gpuProcessor;
        this.texture = texture;
        this.textureView = texture.createView();
        this.width = width;
        this.height = height;
    }


    public static async createFromFloatData(
        gpuProcessor: GpuImageProcessor,
        imageData: Float32ArrayImage
    ): Promise<Image> {
        const device = gpuProcessor.getDevice();

        const texture = device.createTexture({
            label: "Image Texture (rgba32float)",
            size: [imageData.width, imageData.height],
            format: "rgba32float",
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.COPY_SRC,
        });


        device.queue.writeTexture(
            { texture },
            imageData.data,
            {
                bytesPerRow: imageData.width * 4 * 4,
            },
            {
                width: imageData.width,
                height: imageData.height,
            }
        );

        return new Image(gpuProcessor, texture, imageData.width, imageData.height);
    }

    public clone(): Image {
        const device = this.gpuProcessor.getDevice();
        const commandEncoder = device.createCommandEncoder();

        const newTexture = device.createTexture({
            label: "Cloned Image Texture",
            size: [this.width, this.height],
            format: this.texture.format,
            usage: this.texture.usage,
        });

        commandEncoder.copyTextureToTexture(
            { texture: this.texture },
            { texture: newTexture },
            [this.width, this.height]
        );

        device.queue.submit([commandEncoder.finish()]);

        return new Image(this.gpuProcessor, newTexture, this.width, this.height);
    }

    public async readRgbaPixels(): Promise<Float32Array> {
        const device = this.gpuProcessor.getDevice();
        const queue = this.gpuProcessor.getQueue();

        if (this.texture.format !== 'rgba32float') {

            throw new Error(`Reading pixels is currently only supported for 'rgba32float' format, but got '${this.texture.format}'.`);
        }

        const bytesPerPixel = 16;
        const unpaddedBytesPerRow = this.width * bytesPerPixel;
        const alignment = 256;
        const paddedBytesPerRow = Math.ceil(unpaddedBytesPerRow / alignment) * alignment;

        const readbackBuffer = device.createBuffer({
            label: "Image Readback Buffer",
            size: paddedBytesPerRow * this.height,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const commandEncoder = device.createCommandEncoder();
        commandEncoder.copyTextureToBuffer(
            { texture: this.texture },
            { buffer: readbackBuffer, bytesPerRow: paddedBytesPerRow },
            { width: this.width, height: this.height }
        );
        queue.submit([commandEncoder.finish()]);

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const mappedRange = readbackBuffer.getMappedRange();

        const output = new Float32Array(this.width * this.height * 4);
        const source = new Float32Array(mappedRange);


        for (let y = 0; y < this.height; y++) {
            const sourceStart = y * (paddedBytesPerRow / 4);
            const destStart = y * this.width * 4;
            output.set(source.subarray(sourceStart, sourceStart + this.width * 4), destStart);
        }

        readbackBuffer.unmap();
        readbackBuffer.destroy();

        return output;
    }

    public destroy(): void {
        if (this.destroyed) {
            return;
        }

        this.texture.destroy();
        this.destroyed = true;
    }
}

export async function loadPpm(file: File): Promise<Float32ArrayImage> {

    const buffer = await file.arrayBuffer();
    const bytes = new Uint8Array(buffer);

    let offset = 0;

    function readLine(): string {
        let start = offset;
        while (bytes[offset] !== 0x0A) offset++;
        const line = new TextDecoder().decode(bytes.slice(start, offset));
        offset++;
        return line.trim();
    }

    const magic = readLine();
    if (magic !== "P6") {
        throw new Error("PPM format must be P6");
    }

    const sizeLine = readLine();
    const [width, height] = sizeLine.split(" ").map(Number);

    const maxLine = readLine();
    const maxVal = Number(maxLine);
    if (maxVal !== 65535) {
        throw new Error("Only 16bit PPM (max=65535) supported");
    }


    const pixelCount = width * height * 3;
    const data16 = new Uint16Array(pixelCount);

    let p = offset;
    for (let i = 0; i < pixelCount; i++) {
        data16[i] = (bytes[p] << 8) | bytes[p + 1];
        p += 2;
    }

    const floatData = new Float32Array(width * height * 4);

    for (let i = 0, j = 0; i < pixelCount; i += 3, j += 4) {
        floatData[j + 0] = data16[i + 0] / 65535.0;
        floatData[j + 1] = data16[i + 1] / 65535.0;
        floatData[j + 2] = data16[i + 2] / 65535.0;
        floatData[j + 3] = 1.0;
    }

    return { data: floatData, width: width, height: height }
}


export function resizeFloat32RGBALongEdge(
    src: Float32ArrayImage,
    targetLongEdge: number
): Float32ArrayImage {

    const srcWidth = src.width;
    const srcHeight = src.height

    const src_data = src.data;


    let dstWidth: number;
    let dstHeight: number;

    if (srcWidth >= srcHeight) {
        dstWidth = targetLongEdge;
        dstHeight = Math.round(srcHeight * (targetLongEdge / srcWidth));
    } else {
        dstHeight = targetLongEdge;
        dstWidth = Math.round(srcWidth * (targetLongEdge / srcHeight));
    }

    const dst = new Float32Array(dstWidth * dstHeight * 4);

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

            const i00 = (y0 * srcWidth + x0) * 4;
            const i10 = (y0 * srcWidth + x1) * 4;
            const i01 = (y1 * srcWidth + x0) * 4;
            const i11 = (y1 * srcWidth + x1) * 4;

            const di = (y * dstWidth + x) * 4;

            for (let c = 0; c < 4; c++) {
                const c00 = src_data[i00 + c];
                const c10 = src_data[i10 + c];
                const c01 = src_data[i01 + c];
                const c11 = src_data[i11 + c];

                const cx0 = c00 * (1.0 - tx) + c10 * tx;
                const cx1 = c01 * (1.0 - tx) + c11 * tx;

                dst[di + c] = cx0 * (1.0 - ty) + cx1 * ty;
            }
        }
    }

    return {
        data: dst,
        width: dstWidth,
        height: dstHeight,
    };
}


export async function imageBitmapToFloat32RGBA(
    imageBitmap: ImageBitmap
): Promise<Float32ArrayImage> {

    const width = imageBitmap.width;
    const height = imageBitmap.height;

    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext("2d");

    if (!ctx) {
        throw new Error("Failed to get 2D context");
    }

    ctx.drawImage(imageBitmap, 0, 0);

    const imageData = ctx.getImageData(0, 0, width, height);
    const src = imageData.data;

    const dst = new Float32Array(width * height * 4);

    for (let i = 0; i < src.length; i += 4) {


        dst[i + 0] = srgbToLinear(src[i + 0] / 255.0);
        dst[i + 1] = srgbToLinear(src[i + 1] / 255.0);
        dst[i + 2] = srgbToLinear(src[i + 2] / 255.0);


        dst[i + 3] = src[i + 3] / 255.0;
    }

    return {
        data: dst,
        width,
        height,
    };
}

function srgbToLinear(v: number): number {
    if (v <= 0.04045) {
        return v / 12.92;
    }

    return Math.pow((v + 0.055) / 1.055, 2.4);
}