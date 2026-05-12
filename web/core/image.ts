

import type { GpuImageProcessor } from "./gpu_image_processor.ts";

export class Image {
    private gpuProcessor: GpuImageProcessor;
    public readonly texture: GPUTexture;
    public readonly textureView: GPUTextureView;
    public readonly width: number;
    public readonly height: number;

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


    public static async createFromImageBitmap(
        gpuProcessor: GpuImageProcessor,
        imageBitmap: ImageBitmap,
    ): Promise<Image> {
        const device = gpuProcessor.getDevice();


        const tempTexture = device.createTexture({
            label: "Temp Image Texture",
            size: [imageBitmap.width, imageBitmap.height],
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });

        // @ts-ignore
        device.queue.copyExternalImageToTexture(
            { source: imageBitmap },
            { texture: tempTexture },
            [imageBitmap.width, imageBitmap.height]
        );


        const finalTexture = device.createTexture({
            label: "Image Texture (rgba32float)",
            size: [imageBitmap.width, imageBitmap.height],
            format: 'rgba32float',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.COPY_SRC,
        });



        const conversionShader = `
            @group(0) @binding(0) var inputTex: texture_2d<f32>;
            @group(0) @binding(1) var outputTex: texture_storage_2d<rgba32float, write>;

            @compute @workgroup_size(8, 8, 1)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let color_normalized = textureLoad(inputTex, gid.xy, 0);
                textureStore(outputTex, gid.xy, color_normalized);
            }
        `;
        const pipeline = await device.createComputePipelineAsync({
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: conversionShader }), entryPoint: 'main' }
        });
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: tempTexture.createView() },
                { binding: 1, resource: finalTexture.createView() }
            ]
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(imageBitmap.width / 8), Math.ceil(imageBitmap.height / 8));
        passEncoder.end();
        device.queue.submit([commandEncoder.finish()]);


        tempTexture.destroy();

        return new Image(gpuProcessor, finalTexture, imageBitmap.width, imageBitmap.height);
    }

    public static async createFromPpm(
        gpuProcessor: GpuImageProcessor,
        file: File
    ): Promise<Image> {
        const device = gpuProcessor.getDevice();


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


        const texture = device.createTexture({
            label: "PPM Image Texture (rgba32float)",
            size: [width, height],
            format: "rgba32float",
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.COPY_SRC,
        });


        device.queue.writeTexture(
            { texture },
            floatData,
            {
                bytesPerRow: width * 4 * 4,
            },
            {
                width,
                height,
            }
        );

        return new Image(gpuProcessor, texture, width, height);
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
}