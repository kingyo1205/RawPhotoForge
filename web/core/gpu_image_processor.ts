

import { Image } from './image.ts';
import { Mask, CURVE_RESOLUTION } from './photo_editor.ts';


import shaderCode from "./wgpu_shader.wgsl" with { type: "text" };


export interface BaseParams {
    width: number;
    height: number;
    num_masks: number;
}

export interface GpuEditParameters {
    r_gain: number;
    g_gain: number;
    b_gain: number;
    vignette: number;
    lens_distortion: number;
    exposure: number;
    contrast: number;
    shadow: number;
    highlight: number;
    black: number;
    white: number;
}


export class GpuImageProcessor {
    private adapter: GPUAdapter;
    private device: GPUDevice;
    private queue: GPUQueue;
    private pipeline: GPUComputePipeline;
    private bindGroupLayout: GPUBindGroupLayout;


    private constructor(
        adapter: GPUAdapter,
        device: GPUDevice,
        queue: GPUQueue,
        pipeline: GPUComputePipeline,
        bindGroupLayout: GPUBindGroupLayout,
    ) {
        this.adapter = adapter;
        this.device = device;
        this.queue = queue;
        this.pipeline = pipeline;
        this.bindGroupLayout = bindGroupLayout;
    }

    public static async create(): Promise<GpuImageProcessor> {
        if (!navigator.gpu) {
            throw new Error("This browser does not support WebGPU.");
        }

        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) {
            throw new Error("A suitable GPU adapter could not be found.");
        }

        const device = await adapter.requestDevice();
        const queue = device.queue;


        const bindGroupLayout = device.createBindGroupLayout({
            label: "Main Bind Group Layout",
            entries: [

                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "unfilterable-float" },
                },

                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: "write-only", format: "rgba32float" },
                },

                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" },
                },

                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { viewDimension: "2d-array", sampleType: "unfilterable-float" },
                },

                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },

                ...[5, 6, 7, 8].map(binding => ({
                    binding,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" as GPUBufferBindingType },
                }))
            ],
        });

        const pipelineLayout = device.createPipelineLayout({
            label: "Main Pipeline Layout",
            bindGroupLayouts: [bindGroupLayout],
        });

        const shaderModule = device.createShaderModule({
            label: "Shader Module",
            code: shaderCode,
        });

        const pipeline = await device.createComputePipelineAsync({
            label: "Main Compute Pipeline",
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: "main",
            },
        });

        return new GpuImageProcessor(adapter, device, queue, pipeline, bindGroupLayout);
    }

    public getAdapter(): GPUAdapter {
        return this.adapter;
    }

    public getDevice(): GPUDevice {
        return this.device;
    }

    public getQueue(): GPUQueue {
        return this.queue;
    }

    private createStorageBuffer(label: string, data: Float32Array | Int32Array): GPUBuffer {
        const buffer = this.device.createBuffer({
            label,
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });
        const writeArray = data instanceof Float32Array
            ? new Float32Array(buffer.getMappedRange())
            : new Int32Array(buffer.getMappedRange());
        writeArray.set(data);
        buffer.unmap();
        return buffer;
    }


    public async applyAdjustments(
        originalImage: Image,
        masksData: Mask[],
    ): Promise<Image> {
        const { width, height } = originalImage;
        const numMasks = masksData.length;




        const gpuParamsData = new Float32Array(numMasks * 16);
        const allBrightnessLuts = new Int32Array(numMasks * CURVE_RESOLUTION);
        const allHueLuts = new Int32Array(numMasks * CURVE_RESOLUTION);
        const allSatLuts = new Int32Array(numMasks * CURVE_RESOLUTION);
        const allLightLuts = new Int32Array(numMasks * CURVE_RESOLUTION);

        masksData.forEach((mask, i) => {
            const p = mask.editParameters;
            const offset = i * 16;
            gpuParamsData.set([
                1.0 + 0.5 * (p.wb_temperature / 100.0),
                1.0 - 0.25 * (p.wb_tint / 100.0),
                1.0 - 0.5 * (p.wb_temperature / 100.0),
                p.vignette,
                p.lens_distortion,
                p.exposure,
                p.contrast / 100.0,
                p.shadow / 100.0,
                p.highlight / 100.0,
                p.black / 100.0,
                p.white / 100.0,
            ], offset);

            allBrightnessLuts.set(p.brightness_tone_curve, i * CURVE_RESOLUTION);
            allHueLuts.set(p.hue_tone_curve, i * CURVE_RESOLUTION);
            allSatLuts.set(p.saturation_tone_curve, i * CURVE_RESOLUTION);
            allLightLuts.set(p.lightness_tone_curve, i * CURVE_RESOLUTION);
        });


        const baseParamsBuf = this.device.createBuffer({
            label: "Base Params Buffer",
            size: 12,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.queue.writeBuffer(baseParamsBuf, 0, new Uint32Array([width, height, numMasks]));

        const paramBuf = this.createStorageBuffer("Params Storage Buffer", gpuParamsData);
        const brightBuf = this.createStorageBuffer("Brightness LUTs Buffer", allBrightnessLuts);
        const hueBuf = this.createStorageBuffer("Hue LUTs Buffer", allHueLuts);
        const satBuf = this.createStorageBuffer("Saturation LUTs Buffer", allSatLuts);
        const lightBuf = this.createStorageBuffer("Lightness LUTs Buffer", allLightLuts);


        const masksTex = this.device.createTexture({
            label: "Masks Array Texture",
            size: { width, height, depthOrArrayLayers: Math.max(1, numMasks) },
            format: 'r32float',
            dimension: '2d',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });


        const outputTex = this.device.createTexture({
            label: "Output Texture",
            size: { width, height },
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        });


        const bindGroup = this.device.createBindGroup({
            label: "Main Bind Group",
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: originalImage.textureView },
                { binding: 1, resource: outputTex.createView() },
                { binding: 2, resource: { buffer: baseParamsBuf } },
                { binding: 3, resource: masksTex.createView({ dimension: '2d-array' }) },
                { binding: 4, resource: { buffer: paramBuf } },
                { binding: 5, resource: { buffer: brightBuf } },
                { binding: 6, resource: { buffer: hueBuf } },
                { binding: 7, resource: { buffer: satBuf } },
                { binding: 8, resource: { buffer: lightBuf } },
            ],
        });


        const commandEncoder = this.device.createCommandEncoder({ label: "Main Command Encoder" });


        masksData.forEach((mask, i) => {
            commandEncoder.copyTextureToTexture(
                { texture: mask.gpuMask.texture },
                { texture: masksTex, origin: { z: i } },
                { width, height, depthOrArrayLayers: 1 }
            );
        });

        const passEncoder = commandEncoder.beginComputePass({ label: "Main Compute Pass" });
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16), 1);
        passEncoder.end();

        this.queue.submit([commandEncoder.finish()]);


        return new Image(this, outputTex, width, height);
    }
}
