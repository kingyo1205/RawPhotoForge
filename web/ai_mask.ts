import {
    RawImage,
    Sam2Model,
    Sam2Processor,
    Tensor
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

export { RawImage };

export interface SamMask {
    data: Float32Array;
    width: number;
    height: number;
}

const MODEL_ID = "onnx-community/sam2.1-hiera-large-ONNX";

let model: Sam2Model | null = null;
let processor: Sam2Processor | null = null;
let loadingPromise: Promise<void> | null = null;

async function loadModel(onProgress?: (progress: number) => void): Promise<void> {
    model = await Sam2Model.from_pretrained(MODEL_ID, {
        progress_callback: (p: any) => {
            if (typeof p?.progress === "number") {
                onProgress?.(p.progress);
            }
        },
        dtype: {
            vision_encoder: "fp16",
            prompt_encoder_mask_decoder: "fp16",
        },
        device: "webgpu",
    });
    processor = await Sam2Processor.from_pretrained(MODEL_ID);
}

export function ensureModelLoaded(onProgress?: (progress: number) => void): Promise<void> {
    if (model && processor) {
        return Promise.resolve();
    }
    if (!loadingPromise) {
        loadingPromise = loadModel(onProgress).catch((e) => {
            loadingPromise = null;
            model = null;
            processor = null;
            throw e;
        });
    }
    return loadingPromise;
}

export async function generateMask(
    image: InstanceType<typeof RawImage>,
    points: { x: number; y: number }[]
): Promise<SamMask> {
    if (!model || !processor) {
        throw new Error("SAM2 model is not loaded yet. Call ensureModelLoaded() first.");
    }
    if (points.length === 0) {
        throw new Error("No points were provided.");
    }

    const imageProcessed = await processor(image);
    const imageEmbeddings = await model.get_image_embeddings(imageProcessed);
    const reshaped = imageProcessed.reshaped_input_sizes[0]; // [height, width]

    const flatPoints = points
        .map((p) => [p.x * reshaped[1], p.y * reshaped[0]])
        .flat();

    const labels = new Array(points.length).fill(1n);

    const input_points = new Tensor("float32", flatPoints, [1, 1, points.length, 2]);
    const input_labels = new Tensor("int64", labels, [1, 1, points.length]);

    const { pred_masks, iou_scores } = await model({
        ...imageEmbeddings,
        input_points,
        input_labels,
    });

    const masks = await processor.post_process_masks(
        pred_masks,
        imageProcessed.original_sizes,
        imageProcessed.reshaped_input_sizes,
        { binarize: false },
    );

    const scoreData: number[] = Array.from((iou_scores as any).data ?? iou_scores);
    let bestIndex = 0;
    for (let i = 1; i < scoreData.length; i++) {
        if (scoreData[i] > scoreData[bestIndex]) {
            bestIndex = i;
        }
    }
    console.log(masks);

    const maskTensor = masks[0][0][bestIndex];
    const dims = maskTensor.dims as number[];
    const height = dims[dims.length - 2];
    const width = dims[dims.length - 1];

    const mask = Float32Array.from(maskTensor.data as Float32Array);


    return {
        data: mask,
        width,
        height,
    };
}