// core/image.ts
var Image = class _Image {
  gpuProcessor;
  texture;
  textureView;
  width;
  height;
  constructor(gpuProcessor2, texture, width, height) {
    this.gpuProcessor = gpuProcessor2;
    this.texture = texture;
    this.textureView = texture.createView();
    this.width = width;
    this.height = height;
  }
  static async createFromFloatData(gpuProcessor2, imageData) {
    const device = gpuProcessor2.getDevice();
    const texture = device.createTexture({
      label: "Image Texture (rgba32float)",
      size: [
        imageData.width,
        imageData.height
      ],
      format: "rgba32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC
    });
    device.queue.writeTexture({
      texture
    }, imageData.data, {
      bytesPerRow: imageData.width * 4 * 4
    }, {
      width: imageData.width,
      height: imageData.height
    });
    return new _Image(gpuProcessor2, texture, imageData.width, imageData.height);
  }
  clone() {
    const device = this.gpuProcessor.getDevice();
    const commandEncoder = device.createCommandEncoder();
    const newTexture = device.createTexture({
      label: "Cloned Image Texture",
      size: [
        this.width,
        this.height
      ],
      format: this.texture.format,
      usage: this.texture.usage
    });
    commandEncoder.copyTextureToTexture({
      texture: this.texture
    }, {
      texture: newTexture
    }, [
      this.width,
      this.height
    ]);
    device.queue.submit([
      commandEncoder.finish()
    ]);
    return new _Image(this.gpuProcessor, newTexture, this.width, this.height);
  }
  async readRgbaPixels() {
    const device = this.gpuProcessor.getDevice();
    const queue = this.gpuProcessor.getQueue();
    if (this.texture.format !== "rgba32float") {
      throw new Error(`Reading pixels is currently only supported for 'rgba32float' format, but got '${this.texture.format}'.`);
    }
    const bytesPerPixel = 16;
    const unpaddedBytesPerRow = this.width * bytesPerPixel;
    const alignment = 256;
    const paddedBytesPerRow = Math.ceil(unpaddedBytesPerRow / alignment) * alignment;
    const readbackBuffer = device.createBuffer({
      label: "Image Readback Buffer",
      size: paddedBytesPerRow * this.height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyTextureToBuffer({
      texture: this.texture
    }, {
      buffer: readbackBuffer,
      bytesPerRow: paddedBytesPerRow
    }, {
      width: this.width,
      height: this.height
    });
    queue.submit([
      commandEncoder.finish()
    ]);
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
};
async function loadPpm(file) {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let offset = 0;
  function readLine() {
    let start = offset;
    while (bytes[offset] !== 10) offset++;
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
    data16[i] = bytes[p] << 8 | bytes[p + 1];
    p += 2;
  }
  const floatData = new Float32Array(width * height * 4);
  for (let i = 0, j = 0; i < pixelCount; i += 3, j += 4) {
    floatData[j + 0] = data16[i + 0] / 65535;
    floatData[j + 1] = data16[i + 1] / 65535;
    floatData[j + 2] = data16[i + 2] / 65535;
    floatData[j + 3] = 1;
  }
  return {
    data: floatData,
    width,
    height
  };
}
function resizeFloat32RGBALongEdge(src, targetLongEdge) {
  const srcWidth = src.width;
  const srcHeight = src.height;
  const src_data = src.data;
  let dstWidth;
  let dstHeight;
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
        const cx0 = c00 * (1 - tx) + c10 * tx;
        const cx1 = c01 * (1 - tx) + c11 * tx;
        dst[di + c] = cx0 * (1 - ty) + cx1 * ty;
      }
    }
  }
  return {
    data: dst,
    width: dstWidth,
    height: dstHeight
  };
}
async function imageBitmapToFloat32RGBA(imageBitmap) {
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
    dst[i + 0] = srgbToLinear(src[i + 0] / 255);
    dst[i + 1] = srgbToLinear(src[i + 1] / 255);
    dst[i + 2] = srgbToLinear(src[i + 2] / 255);
    dst[i + 3] = src[i + 3] / 255;
  }
  return {
    data: dst,
    width,
    height
  };
}
function srgbToLinear(v) {
  if (v <= 0.04045) {
    return v / 12.92;
  }
  return Math.pow((v + 0.055) / 1.055, 2.4);
}

// core/interpolation.ts
function pchipInterpolate(xPts, yPts, xEval) {
  if (xPts.length !== yPts.length) {
    throw new Error(`Input arrays must have the same length: x has ${xPts.length}, y has ${yPts.length}`);
  }
  if (xPts.length < 2) {
    throw new Error(`At least two points are required for interpolation, but got ${xPts.length}`);
  }
  const n = xPts.length;
  const yEval = new Float32Array(xEval.length);
  const h = new Float32Array(n - 1);
  const del = new Float32Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    const h_i = xPts[i + 1] - xPts[i];
    if (h_i <= 0) {
      throw new Error(`x_pts must be strictly increasing, but found a non-increasing value at index ${i}`);
    }
    h[i] = h_i;
    del[i] = (yPts[i + 1] - yPts[i]) / h_i;
  }
  const slopes = new Float32Array(n);
  slopes[0] = del[0];
  slopes[n - 1] = del[n - 2];
  for (let i = 1; i < n - 1; i++) {
    if (del[i - 1] * del[i] <= 0) {
      slopes[i] = 0;
    } else {
      const w1 = 2 * h[i] + h[i - 1];
      const w2 = h[i] + 2 * h[i - 1];
      slopes[i] = (w1 + w2) / (w1 / del[i - 1] + w2 / del[i]);
    }
  }
  for (let k = 0; k < xEval.length; k++) {
    const x = xEval[k];
    if (x <= xPts[0]) {
      yEval[k] = yPts[0];
      continue;
    }
    if (x >= xPts[n - 1]) {
      yEval[k] = yPts[n - 1];
      continue;
    }
    let i = Array.prototype.findIndex.call(xPts, (p) => p > x);
    if (i === -1) {
      i = n - 1;
    }
    i = Math.max(0, i - 1);
    i = Math.min(i, n - 2);
    const h_val = h[i];
    const t = (x - xPts[i]) / h_val;
    const t2 = t * t;
    const t3 = t2 * t;
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;
    const y0 = yPts[i];
    const y1 = yPts[i + 1];
    const m0 = slopes[i];
    const m1 = slopes[i + 1];
    yEval[k] = h00 * y0 + h10 * h_val * m0 + h01 * y1 + h11 * h_val * m1;
  }
  return yEval;
}

// core/photo_editor.ts
var CURVE_RESOLUTION = 65536;
var PhotoEditor = class _PhotoEditor {
  gpuProcessor;
  image;
  originalImage;
  masks = [];
  constructor(gpuProcessor2, image) {
    this.gpuProcessor = gpuProcessor2;
    this.image = image;
    this.originalImage = image.clone();
    const mainMask = this.createGpuMask(image.width, image.height);
    this.masks.push({
      name: "main",
      gpuMask: mainMask,
      editParameters: createDefaultEditParameters()
    });
  }
  static async create(gpuProcessor2, imageData) {
    const image = await Image.createFromFloatData(gpuProcessor2, imageData);
    return new _PhotoEditor(gpuProcessor2, image);
  }
  createGpuMask(width, height, data2) {
    const device = this.gpuProcessor.getDevice();
    const texture = device.createTexture({
      label: "Mask Texture",
      size: {
        width,
        height
      },
      format: "r32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC
    });
    const maskData = data2 ?? new Float32Array(width * height).fill(1);
    this.gpuProcessor.getQueue().writeTexture({
      texture
    }, maskData.buffer, {
      bytesPerRow: width * 4
    }, {
      width,
      height
    });
    return {
      texture,
      view: texture.createView()
    };
  }
  async save() {
    const rgbaPixels = await this.image.readRgbaPixels();
    const clampedPixels = new Uint8ClampedArray(rgbaPixels.length);
    for (let i = 0; i < rgbaPixels.length; i++) {
      clampedPixels[i] = rgbaPixels[i] * 255;
    }
    const imageData = new ImageData(clampedPixels, this.image.width, this.image.height);
    return await createImageBitmap(imageData);
  }
  reset() {
    this.image = this.originalImage.clone();
    this.masks = this.masks.filter((m) => m.name === "main");
    const mainMask = this.masks.find((m) => m.name === "main");
    if (mainMask) {
      mainMask.editParameters = createDefaultEditParameters();
    }
  }
  getAdjustmentSet(maskName) {
    const name = maskName ?? "main";
    const mask = this.masks.find((m) => m.name === name);
    if (!mask) {
      throw new Error(`The specified mask '${name}' does not exist.`);
    }
    return mask.editParameters;
  }
  setWhitebalance(temperature, tint, maskName) {
    const adjustments = this.getAdjustmentSet(maskName);
    adjustments.wb_temperature = clamp(temperature, -100, 100);
    adjustments.wb_tint = clamp(tint, -100, 100);
  }
  setVignette(value) {
    this.getAdjustmentSet("main").vignette = clamp(value, -100, 100);
  }
  setLensDistortionCorrection(value) {
    this.getAdjustmentSet("main").lens_distortion = clamp(value, -100, 100);
  }
  setTone(exposure, contrast, shadow, highlight, black, white, maskName) {
    const adjustments = this.getAdjustmentSet(maskName);
    adjustments.exposure = clamp(exposure, -10, 10);
    adjustments.contrast = clamp(contrast, -100, 100);
    adjustments.shadow = clamp(shadow, -100, 100);
    adjustments.highlight = clamp(highlight, -100, 100);
    adjustments.black = clamp(black, -100, 100);
    adjustments.white = clamp(white, -100, 100);
  }
  _setCurve(curveType, curve, controlPointsX, controlPointsY, maskName) {
    if (curve === void 0 && controlPointsX === void 0) {
      throw new Error("You must provide either a full curve or control points.");
    }
    let finalCurve;
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
      finalCurve = new Int32Array(interpolated.map((v) => clamp(v, 0, 65535)));
    }
    this.getAdjustmentSet(maskName)[curveType] = finalCurve;
  }
  setBrightnessToneCurve(curve, controlPointsX, controlPointsY, maskName) {
    this._setCurve("brightness_tone_curve", curve, controlPointsX, controlPointsY, maskName);
  }
  setOklchHueCurve(curve, controlPointsX, controlPointsY, maskName) {
    this._setCurve("hue_tone_curve", curve, controlPointsX, controlPointsY, maskName);
  }
  setOklchSaturationCurve(curve, controlPointsX, controlPointsY, maskName) {
    this._setCurve("saturation_tone_curve", curve, controlPointsX, controlPointsY, maskName);
  }
  setOklchLightnessCurve(curve, controlPointsX, controlPointsY, maskName) {
    this._setCurve("lightness_tone_curve", curve, controlPointsX, controlPointsY, maskName);
  }
  addMask(name, maskData) {
    const gpuMask = this.createGpuMask(this.originalImage.width, this.originalImage.height, maskData);
    this.masks.push({
      name,
      gpuMask,
      editParameters: createDefaultEditParameters()
    });
  }
  removeMask(name) {
    if (name === "main") {
      throw new Error("The main mask cannot be deleted.");
    }
    this.masks = this.masks.filter((mask) => mask.name !== name);
  }
  async applyAdjustments() {
    this.image = await this.gpuProcessor.applyAdjustments(this.originalImage, this.masks);
  }
};
function clamp(num, min, max) {
  return Math.min(Math.max(num, min), max);
}
function createDefaultEditParameters() {
  const resolutionRange = new Int32Array(CURVE_RESOLUTION).map((_, i) => i);
  return {
    exposure: 0,
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
    lightness_tone_curve: new Int32Array(CURVE_RESOLUTION).fill(32767)
  };
}

// core/wgpu_shader.wgsl
var wgpu_shader_default = "// wgpu_shader.wgsl\n\n//--------------------------------------------------------------------------------\n// Bindings\n//--------------------------------------------------------------------------------\n@group(0) @binding(0) var image_in: texture_2d<f32>;\n@group(0) @binding(1) var image_out: texture_storage_2d<rgba32float, write>;\n@group(0) @binding(2) var<uniform> base_params: BaseParams;\n@group(0) @binding(3) var masks_tex: texture_2d_array<f32>;\n@group(0) @binding(4) var<storage, read> masks_params: array<GpuEditParameters>;\n\n@group(0) @binding(5) var<storage, read> brightness_curve: array<i32>;\n@group(0) @binding(6) var<storage, read> hue_curve: array<i32>;\n@group(0) @binding(7) var<storage, read> saturation_curve: array<i32>;\n@group(0) @binding(8) var<storage, read> lightness_curve: array<i32>;\n\nstruct BaseParams {\n    width: u32,\n    height: u32,\n    num_masks: u32\n};\n\nstruct GpuEditParameters {\n    r_gain: f32,\n    g_gain: f32,\n    b_gain: f32,\n    vignette: f32,\n    lens_distortion: f32,\n    exposure: f32,\n    contrast: f32,\n    shadow: f32,\n    highlight: f32,\n    black: f32,\n    white: f32,\n};\n\n\n\n\n//--------------------------------------------------------------------------------\n// Color Space Conversions (vec4)\n//--------------------------------------------------------------------------------\n\nconst M1 = mat3x3<f32>(\n    0.4122214708, 0.2119034982, 0.0883024619,\n    0.5363325363, 0.6806995451, 0.2817188376,\n    0.0514459929, 0.1073969566, 0.6299787005\n);\n\nconst M1_INV = mat3x3<f32>(\n    4.0767416621, -1.2684380046, -0.0041960863,\n    -3.3077115913, 2.6097574011, -0.7034186147,\n    0.2309699292, -0.3413193965, 1.7076147010\n);\n\nconst M2 = mat3x3<f32>(\n    0.2104542553, 1.9779984951, 0.0259040371,\n    0.7936177850, -2.4285922050, 0.7827717662,\n    -0.0040720468, 0.4505937099, -0.8086757660\n);\n\nconst M2_INV = mat3x3<f32>(\n    1.0, 1.0, 1.0,\n    0.3963377774, -0.1055613458, -0.089484177,\n    0.2158037573, -0.0638541728, -1.2914855480\n);\n\nfn linear_srgb_to_oklch(c: vec4<f32>) -> vec4<f32> {\n    let lms = M1 * c.rgb;\n    // pow(x, 1/3) with protection against negative/zero\n    let lms_cbrt = pow(max(lms, vec3<f32>(0.0)), vec3<f32>(1.0 / 3.0));\n    let oklab = M2 * lms_cbrt;\n    \n    let L = oklab.x;\n    let C = sqrt(oklab.y * oklab.y + oklab.z * oklab.z);\n    var h = atan2(oklab.z, oklab.y) / (2.0 * 3.14159265359);\n    if (h < 0.0) { h += 1.0; }\n    return vec4<f32>(L, C, h, 1.0);\n}\n\nfn oklch_to_linear_srgb(c: vec4<f32>) -> vec4<f32> {\n    let h = c.z * 2.0 * 3.14159265359;\n    let oklab = vec3<f32>(c.x, c.y * cos(h), c.y * sin(h));\n    \n    let lms_cbrt = M2_INV * oklab;\n    let lms = lms_cbrt * lms_cbrt * lms_cbrt;\n    return vec4<f32>(M1_INV * lms, 1.0);\n}\nfn srgb_to_linear(c: vec4<f32>) -> vec4<f32> {\n    let rgb = c.rgb;\n    let linear_rgb = select(\n        pow((rgb + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)),\n        rgb / 12.92,\n        rgb <= vec3<f32>(0.04045)\n    );\n    return vec4<f32>(linear_rgb, 1.0);\n}\n\nfn linear_to_srgb(c: vec4<f32>) -> vec4<f32> {\n    let rgb = c.rgb;\n    let srgb = select(\n        1.055 * pow(rgb, vec3<f32>(1.0 / 2.4)) - 0.055,\n        rgb * 12.92,\n        rgb <= vec3<f32>(0.0031308)\n    );\n    return vec4<f32>(srgb, 1.0);\n}\n\n//--------------------------------------------------------------------------------\n// Effects\n//--------------------------------------------------------------------------------\n\nfn lens_distortion_sample(xy: vec2<i32>, distortion: f32) -> vec4<f32> {\n    let strength = -0.5 * (distortion / 100.0);\n\n    let w_u: u32 = base_params.width;\n    let h_u: u32 = base_params.height;\n\n    let w: f32 = f32(w_u);\n    let h: f32 = f32(h_u);\n\n    // \u305D\u306E\u307E\u307E(\u6B6A\u307F\u306A\u3057)\u306A\u3089\u6700\u901F\u3067\u8FD4\u3059\n    if (strength == 0.0) {\n        return textureLoad(image_in, xy, 0);\n    }\n\n    // UV(0..1)\n    let uv = vec2<f32>(f32(xy.x) / w, f32(xy.y) / h);\n\n    // \u6B6A\u307F\u88DC\u6B63\n    let centered_uv = uv - 0.5;\n    let aspect = w / h;\n\n    var corrected_uv = centered_uv * vec2<f32>(aspect, 1.0);\n\n    let r2 = dot(corrected_uv, corrected_uv);\n    let distorted = corrected_uv / (1.0 + strength * r2);\n    let final_uv = (distorted / vec2<f32>(aspect, 1.0)) + 0.5;\n\n    // \u7BC4\u56F2\u5916\u306F\u9ED2\n    if (any(final_uv < vec2<f32>(0.0)) || any(final_uv > vec2<f32>(1.0))) {\n        return vec4<f32>(0.0, 0.0, 0.0, 1.0);\n    }\n\n    let px = final_uv.x * (w - 1.0);\n    let py = final_uv.y * (h - 1.0);\n\n    let x0_f = floor(px);\n    let y0_f = floor(py);\n\n    let x0_i: i32 = i32(x0_f);\n    let y0_i: i32 = i32(y0_f);\n\n\n    let x1_i: i32 = min(x0_i + 1, i32(w_u) - 1);\n    let y1_i: i32 = min(y0_i + 1, i32(h_u) - 1);\n\n    let tx: f32 = px - x0_f;\n    let ty: f32 = py - y0_f;\n\n\n    let c00 = textureLoad(image_in, vec2<i32>(x0_i, y0_i), 0);\n    let c10 = textureLoad(image_in, vec2<i32>(x1_i, y0_i), 0);\n    let c01 = textureLoad(image_in, vec2<i32>(x0_i, y1_i), 0);\n    let c11 = textureLoad(image_in, vec2<i32>(x1_i, y1_i), 0);\n\n\n    let cx0 = mix(c00, c10, tx);\n    let cx1 = mix(c01, c11, tx);\n    return mix(cx0, cx1, ty);\n}\n\nfn vignette(rgb_vec4: vec4<f32>, vignette_value: f32, xy: vec2<i32>) -> vec4<f32>{\n    let vign_strength = (-f32(vignette_value) / 100.0) * 2.0;\n    if (vign_strength != 0.0) {\n        let coord = (vec2<f32>(xy) / vec2<f32>(f32(base_params.width), f32(base_params.height)) - 0.5) * 2.0;\n        let dist = length(coord);\n        let falloff = pow(clamp((dist - 0.25) / 0.75, 0.0, 1.0), 4.0);\n        let rgb_vec4 = vec4<f32>(rgb_vec4.rgb * clamp(1.0 - (vign_strength * falloff), 0.0, 2.0), 1.0);\n\n        return rgb_vec4;\n    } else {\n        return rgb_vec4;\n    }\n} \n\n\n//--------------------------------------------------------------------------------\n// Curve\n//--------------------------------------------------------------------------------\n\nfn lut_fetch(which: u32, mask_index: u32, idx: u32) -> u32 {\n    let base = mask_index * 65536u + idx;\n\n    switch(which) {\n        case 0u: { return u32(clamp(brightness_curve[base], 0, 65535)); }\n        case 1u: { return u32(clamp(hue_curve[base], 0, 65535)); }\n        case 2u: { return u32(clamp(saturation_curve[base], 0, 65535)); }\n        case 3u: { return u32(clamp(lightness_curve[base], 0, 65535)); }\n        default  { return 0; }\n    }\n}\n\n//--------------------------------------------------------------------------------\n// Tone\n//--------------------------------------------------------------------------------\n\nfn tone(\n    rgb_vec4: vec4<f32>,\n    exposure: f32,   // -6.0 ~ +6.0 (EV)\n    contrast: f32,   // -1.0 ~ +1.0\n    shadow: f32,     // -1.0 ~ +1.0\n    highlight: f32,  // -1.0 ~ +1.0\n    black: f32,      // -1.0 ~ +1.0\n    white: f32,      // -1.0 ~ +1.0\n    xy: vec2<i32>\n) -> vec4<f32> {\n\n    var color = rgb_vec4.rgb;\n\n    // --- \u9732\u51FA(EV) ---\n    let exposure_mul = pow(2.0, exposure);\n    color *= exposure_mul;\n\n    // \u8F1D\u5EA6(\u30EA\u30CB\u30A2)\n    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));\n\n    // --- \u30B7\u30E3\u30C9\u30A6/\u30CF\u30A4\u30E9\u30A4\u30C8\u30DE\u30B9\u30AF ---\n    let shadow_mask = clamp(1.0 - luma, 0.0, 1.0);\n    // highlight: \u660E\u90E8\u307B\u30691\u306B\u8FD1\u3044\n    let highlight_mask = clamp(luma, 0.0, 1.0);\n\n    // --- \u30B7\u30E3\u30C9\u30A6\u88DC\u6B63 ---\n    let shadow_gain = 1.0 + shadow * shadow_mask;\n    color *= shadow_gain;\n\n    // --- \u30CF\u30A4\u30E9\u30A4\u30C8\u88DC\u6B63 ---\n    let highlight_gain = 1.0 + highlight * highlight_mask;\n    color *= highlight_gain;\n\n    // --- \u30D6\u30E9\u30C3\u30AF ---\n    if (black != 0.0) {\n        let t = clamp(luma, 0.0, 1.0);\n        let black_mask = pow(1.0 - t, 2.0);\n        color += black * black_mask;\n    }\n\n    // --- \u30DB\u30EF\u30A4\u30C8 ---\n    if (white != 0.0) {\n        let t = clamp(luma, 0.0, 1.0);\n        let white_mask = pow(t, 2.0);\n        color += white * white_mask;\n    }\n\n    // --- \u30B3\u30F3\u30C8\u30E9\u30B9\u30C8 ---\n    if (contrast != 0.0) {\n        let pivot = 0.5;\n        let c = 1.0 + contrast;\n\n        color = (color - vec3<f32>(pivot)) * c + vec3<f32>(pivot);\n    }\n\n    // --- \u30AF\u30EA\u30C3\u30D7 ---\n    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));\n\n    return vec4<f32>(color, rgb_vec4.a);\n}\n\n//--------------------------------------------------------------------------------\n// Main\n//--------------------------------------------------------------------------------\n\n@compute @workgroup_size(16, 16, 1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let xy = vec2<i32>(gid.xy);\n    if (gid.x >= base_params.width || gid.y >= base_params.height) { return; }\n\n    let main_p = masks_params[0];\n\n    // Lens Distortion (Main only)\n    var rgb_vec4 = lens_distortion_sample(xy, main_p.lens_distortion);\n    \n\n    // Vignette (Main only)\n    rgb_vec4 = vignette(rgb_vec4, main_p.vignette, xy);\n\n    // Per-mask Linear RGB adjustments\n    for (var mask_index = 0u; mask_index < base_params.num_masks; mask_index++) {\n        let mask_value = textureLoad(masks_tex, xy, mask_index, 0).r;\n        if (mask_value != 1.0) { continue; }\n\n        let p = masks_params[mask_index];\n        \n        // White Balance\n        var r_f32 = rgb_vec4.r * p.r_gain;\n        var g_f32 = rgb_vec4.g * p.g_gain;\n        var b_f32 = rgb_vec4.b * p.b_gain;\n\n\n        // tone\n        rgb_vec4 = tone(vec4<f32>(r_f32, g_f32, b_f32, 1.0), p.exposure, p.contrast, p.shadow, p.highlight, p.black, p.white, xy);\n\n\n        var r_u32 = u32(rgb_vec4.r * 65535);\n        var g_u32 = u32(rgb_vec4.g * 65535);\n        var b_u32 = u32(rgb_vec4.b * 65535);\n        \n\n        \n        \n        // Curves\n        // brightness\n        r_u32 = lut_fetch(0u, mask_index, r_u32);\n        g_u32 = lut_fetch(0u, mask_index, g_u32);\n        b_u32 = lut_fetch(0u, mask_index, b_u32);\n\n        r_f32 = f32(r_u32) / 65535.0;\n        g_f32 = f32(g_u32) / 65535.0;\n        b_f32 = f32(b_u32) / 65535.0;\n\n        rgb_vec4 = vec4<f32>(r_f32, g_f32, b_f32, 1.0);\n    }\n\n    // Per-mask OKLCH adjustments\n    var oklch_vec4 = linear_srgb_to_oklch(rgb_vec4);\n    for (var mask_index = 0u; mask_index < base_params.num_masks; mask_index++) {\n        let mask_value = textureLoad(masks_tex, xy, mask_index, 0).r;\n        if (mask_value != 1.0) { continue; }\n\n        var l_f32 = oklch_vec4.x;\n        var c_f32 = oklch_vec4.y;\n        var h_f32 = oklch_vec4.z;\n\n        var l_u32 = u32(l_f32 * 65535);\n        var c_u32 = u32(c_f32 * 65535);\n        var h_u32 = u32(h_f32 * 65535);\n        \n        let new_hue = lut_fetch(1u, mask_index, h_u32);\n        let saturation_gain = lut_fetch(2u, mask_index, h_u32);\n        let lightness_gain = lut_fetch(3u, mask_index, h_u32);\n\n        oklch_vec4.z = f32(new_hue) / 65535.0;\n        oklch_vec4.y = c_f32 * (f32(saturation_gain) / 32767.5);\n        oklch_vec4.x = l_f32 * (f32(lightness_gain) / 32767.5);\n    }\n    rgb_vec4 = oklch_to_linear_srgb(oklch_vec4);\n\n    // Final Output\n    let out_col = linear_to_srgb(rgb_vec4);\n    textureStore(image_out, xy, clamp(out_col, vec4(0.0), vec4(1.0)));\n}\n";

// core/gpu_image_processor.ts
var GpuImageProcessor = class _GpuImageProcessor {
  adapter;
  device;
  queue;
  pipeline;
  bindGroupLayout;
  constructor(adapter, device, queue, pipeline, bindGroupLayout) {
    this.adapter = adapter;
    this.device = device;
    this.queue = queue;
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;
  }
  static async create() {
    if (!navigator.gpu) {
      throw new Error("This browser does not support WebGPU.");
    }
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance"
    });
    if (!adapter) {
      throw new Error("A suitable GPU adapter could not be found.");
    }
    const device = await adapter.requestDevice({
      requiredLimits: limitsToRecord(adapter.limits)
    });
    const queue = device.queue;
    const bindGroupLayout = device.createBindGroupLayout({
      label: "Main Bind Group Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          texture: {
            sampleType: "unfilterable-float"
          }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            access: "write-only",
            format: "rgba32float"
          }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "uniform"
          }
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          texture: {
            viewDimension: "2d-array",
            sampleType: "unfilterable-float"
          }
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "read-only-storage"
          }
        },
        ...[
          5,
          6,
          7,
          8
        ].map((binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "read-only-storage"
          }
        }))
      ]
    });
    const pipelineLayout = device.createPipelineLayout({
      label: "Main Pipeline Layout",
      bindGroupLayouts: [
        bindGroupLayout
      ]
    });
    const shaderModule = device.createShaderModule({
      label: "Shader Module",
      code: wgpu_shader_default
    });
    const pipeline = await device.createComputePipelineAsync({
      label: "Main Compute Pipeline",
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "main"
      }
    });
    return new _GpuImageProcessor(adapter, device, queue, pipeline, bindGroupLayout);
  }
  getAdapter() {
    return this.adapter;
  }
  getDevice() {
    return this.device;
  }
  getQueue() {
    return this.queue;
  }
  createStorageBuffer(label, data2) {
    const buffer = this.device.createBuffer({
      label,
      size: data2.byteLength,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    const writeArray = data2 instanceof Float32Array ? new Float32Array(buffer.getMappedRange()) : new Int32Array(buffer.getMappedRange());
    writeArray.set(data2);
    buffer.unmap();
    return buffer;
  }
  async applyAdjustments(originalImage, masksData) {
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
        1 + 0.5 * (p.wb_temperature / 100),
        1 - 0.25 * (p.wb_tint / 100),
        1 - 0.5 * (p.wb_temperature / 100),
        p.vignette,
        p.lens_distortion,
        p.exposure,
        p.contrast / 100,
        p.shadow / 100,
        p.highlight / 100,
        p.black / 100,
        p.white / 100
      ], offset);
      allBrightnessLuts.set(p.brightness_tone_curve, i * CURVE_RESOLUTION);
      allHueLuts.set(p.hue_tone_curve, i * CURVE_RESOLUTION);
      allSatLuts.set(p.saturation_tone_curve, i * CURVE_RESOLUTION);
      allLightLuts.set(p.lightness_tone_curve, i * CURVE_RESOLUTION);
    });
    const baseParamsBuf = this.device.createBuffer({
      label: "Base Params Buffer",
      size: 12,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.queue.writeBuffer(baseParamsBuf, 0, new Uint32Array([
      width,
      height,
      numMasks
    ]));
    const paramBuf = this.createStorageBuffer("Params Storage Buffer", gpuParamsData);
    const brightBuf = this.createStorageBuffer("Brightness LUTs Buffer", allBrightnessLuts);
    const hueBuf = this.createStorageBuffer("Hue LUTs Buffer", allHueLuts);
    const satBuf = this.createStorageBuffer("Saturation LUTs Buffer", allSatLuts);
    const lightBuf = this.createStorageBuffer("Lightness LUTs Buffer", allLightLuts);
    const masksTex = this.device.createTexture({
      label: "Masks Array Texture",
      size: {
        width,
        height,
        depthOrArrayLayers: Math.max(1, numMasks)
      },
      format: "r32float",
      dimension: "2d",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });
    const outputTex = this.device.createTexture({
      label: "Output Texture",
      size: {
        width,
        height
      },
      format: "rgba32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    });
    const bindGroup = this.device.createBindGroup({
      label: "Main Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: originalImage.textureView
        },
        {
          binding: 1,
          resource: outputTex.createView()
        },
        {
          binding: 2,
          resource: {
            buffer: baseParamsBuf
          }
        },
        {
          binding: 3,
          resource: masksTex.createView({
            dimension: "2d-array"
          })
        },
        {
          binding: 4,
          resource: {
            buffer: paramBuf
          }
        },
        {
          binding: 5,
          resource: {
            buffer: brightBuf
          }
        },
        {
          binding: 6,
          resource: {
            buffer: hueBuf
          }
        },
        {
          binding: 7,
          resource: {
            buffer: satBuf
          }
        },
        {
          binding: 8,
          resource: {
            buffer: lightBuf
          }
        }
      ]
    });
    const commandEncoder = this.device.createCommandEncoder({
      label: "Main Command Encoder"
    });
    masksData.forEach((mask, i) => {
      commandEncoder.copyTextureToTexture({
        texture: mask.gpuMask.texture
      }, {
        texture: masksTex,
        origin: {
          z: i
        }
      }, {
        width,
        height,
        depthOrArrayLayers: 1
      });
    });
    const passEncoder = commandEncoder.beginComputePass({
      label: "Main Compute Pass"
    });
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16), 1);
    passEncoder.end();
    this.queue.submit([
      commandEncoder.finish()
    ]);
    return new Image(this, outputTex, width, height);
  }
};
function limitsToRecord(limits) {
  const result = {};
  for (const key in limits) {
    const value = limits[key];
    if (typeof value === "number") {
      result[key] = value;
    }
  }
  return result;
}

// tone_curve_editor.ts
var CurveMode = /* @__PURE__ */ function(CurveMode2) {
  CurveMode2[CurveMode2["BRIGHTNESS"] = 0] = "BRIGHTNESS";
  CurveMode2[CurveMode2["HUE"] = 1] = "HUE";
  CurveMode2[CurveMode2["SATURATION"] = 2] = "SATURATION";
  CurveMode2[CurveMode2["LIGHTNESS"] = 3] = "LIGHTNESS";
  return CurveMode2;
}({});
var POINT_RADIUS = 8;
var ToneCurveEditor = class {
  container;
  mode;
  points = [];
  canvas;
  ctx;
  draggingIndex = -1;
  // タッチ操作用
  lastTap = 0;
  touchIdentifier = null;
  hasMoved = false;
  dispatchCurveChange;
  dispatchDragStart;
  dispatchDragEnd;
  constructor(containerId, mode, onCurveChange, onDragStart2, onDragEnd2) {
    this.container = document.getElementById(containerId);
    this.mode = mode;
    this.dispatchCurveChange = () => onCurveChange(this.points);
    this.dispatchDragStart = onDragStart2;
    this.dispatchDragEnd = onDragEnd2;
    this.canvas = document.createElement("canvas");
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.initializePoints();
    this.setupCanvas();
    this.addEventListeners();
    this.draw();
  }
  setBackground(imagePath) {
    this.container.style.backgroundImage = `url(${imagePath})`;
  }
  initializePoints() {
    if (this.mode === CurveMode.BRIGHTNESS || this.mode === CurveMode.HUE) {
      this.points = [
        {
          x: 0,
          y: 0
        },
        {
          x: 1,
          y: 1
        }
      ];
    } else if (this.mode === CurveMode.SATURATION || this.mode === CurveMode.LIGHTNESS) {
      this.points = [
        {
          x: 0,
          y: 1
        },
        {
          x: 1,
          y: 1
        }
      ];
    }
    this.draw();
  }
  setupCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.container.getBoundingClientRect();
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
  }
  addEventListeners() {
    this.canvas.addEventListener("mousedown", this.onMouseDown.bind(this));
    this.canvas.addEventListener("mousemove", this.onMouseMove.bind(this));
    this.canvas.addEventListener("mouseup", this.onPointerUp.bind(this));
    this.canvas.addEventListener("mouseleave", this.onPointerUp.bind(this));
    this.canvas.addEventListener("dblclick", this.onDoubleClick.bind(this));
    this.canvas.addEventListener("touchstart", this.onTouchStart.bind(this), {
      passive: false
    });
    this.canvas.addEventListener("touchmove", this.onTouchMove.bind(this), {
      passive: false
    });
    this.canvas.addEventListener("touchend", this.onTouchEnd.bind(this));
    this.canvas.addEventListener("touchcancel", this.onPointerUp.bind(this));
    this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  }
  getYRange() {
    if (this.mode === CurveMode.SATURATION || this.mode === CurveMode.LIGHTNESS) {
      return {
        min: 0,
        max: 2
      };
    }
    return {
      min: 0,
      max: 1
    };
  }
  toScreen(p) {
    const rect = this.canvas.getBoundingClientRect();
    const yRange = this.getYRange();
    return {
      x: p.x * rect.width,
      y: (1 - (p.y - yRange.min) / (yRange.max - yRange.min)) * rect.height
    };
  }
  toCurve(p) {
    const rect = this.canvas.getBoundingClientRect();
    const yRange = this.getYRange();
    const yNorm = Math.max(0, Math.min(1, 1 - p.y / rect.height));
    return {
      x: Math.max(0, Math.min(1, p.x / rect.width)),
      y: yNorm * (yRange.max - yRange.min) + yRange.min
    };
  }
  getPointerPosition(event) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
  }
  findPoint(pos) {
    for (let i = 0; i < this.points.length; i++) {
      const sp = this.toScreen(this.points[i]);
      const distance = Math.sqrt(Math.pow(sp.x - pos.x, 2) + Math.pow(sp.y - pos.y, 2));
      if (distance <= POINT_RADIUS * 1.5) {
        return i;
      }
    }
    return -1;
  }
  onPointerDown(pos, isRightClick = false) {
    if (isRightClick) {
      this.deletePointAt(pos);
      return;
    }
    const idx = this.findPoint(pos);
    if (idx !== -1) {
      this.draggingIndex = idx;
    } else {
      const p = this.toCurve(pos);
      let insertAt = this.points.findIndex((pt) => pt.x > p.x);
      if (insertAt === -1) {
        insertAt = this.points.length;
      }
      this.points.splice(insertAt, 0, p);
      this.draggingIndex = insertAt;
    }
    this.dispatchDragStart();
    this.emitCurveChange();
  }
  onPointerMove(pos) {
    if (this.draggingIndex === -1) return;
    this.hasMoved = true;
    const p = this.toCurve(pos);
    const minX = this.draggingIndex > 0 ? this.points[this.draggingIndex - 1].x + 1e-3 : 0;
    const maxX = this.draggingIndex < this.points.length - 1 ? this.points[this.draggingIndex + 1].x - 1e-3 : 1;
    p.x = Math.max(minX, Math.min(maxX, p.x));
    if (this.draggingIndex === 0) p.x = 0;
    if (this.draggingIndex === this.points.length - 1) p.x = 1;
    const yRange = this.getYRange();
    p.y = Math.max(yRange.min, Math.min(yRange.max, p.y));
    this.points[this.draggingIndex] = p;
    this.emitCurveChange();
  }
  onPointerUp() {
    if (this.draggingIndex !== -1) {
      this.draggingIndex = -1;
      this.dispatchDragEnd();
    }
    this.touchIdentifier = null;
    this.hasMoved = false;
  }
  deletePointAt(pos) {
    const idx = this.findPoint(pos);
    if (idx > 0 && idx < this.points.length - 1) {
      this.points.splice(idx, 1);
      this.emitCurveChange();
    }
  }
  onMouseDown(event) {
    const pos = this.getPointerPosition(event);
    this.onPointerDown(pos, event.button === 2);
  }
  onMouseMove(event) {
    if (event.buttons !== 1) return;
    const pos = this.getPointerPosition(event);
    this.onPointerMove(pos);
  }
  onDoubleClick(event) {
    const pos = this.getPointerPosition(event);
    this.deletePointAt(pos);
  }
  onTouchStart(event) {
    event.preventDefault();
    if (this.touchIdentifier !== null) return;
    const touch = event.changedTouches[0];
    this.touchIdentifier = touch.identifier;
    const pos = this.getPointerPosition(touch);
    const currentTime = (/* @__PURE__ */ new Date()).getTime();
    const tapLength = currentTime - this.lastTap;
    if (tapLength < 300 && tapLength > 0) {
      this.deletePointAt(pos);
      this.lastTap = 0;
    } else {
      this.onPointerDown(pos);
    }
    this.lastTap = currentTime;
  }
  onTouchMove(event) {
    event.preventDefault();
    if (this.draggingIndex === -1) return;
    const touch = Array.from(event.changedTouches).find((t) => t.identifier === this.touchIdentifier);
    if (!touch) return;
    const pos = this.getPointerPosition(touch);
    this.onPointerMove(pos);
  }
  onTouchEnd(event) {
    event.preventDefault();
    const touch = Array.from(event.changedTouches).find((t) => t.identifier === this.touchIdentifier);
    if (!touch) return;
    this.onPointerUp();
  }
  emitCurveChange() {
    this.draw();
    this.dispatchCurveChange();
  }
  sampleCurve(n) {
    const xPts = this.points.map((p) => p.x);
    const yPts = this.points.map((p) => p.y);
    const xEval = new Float32Array(n).map((_, i) => i / (n - 1));
    return pchipInterpolate(xPts, yPts, xEval);
  }
  draw() {
    requestAnimationFrame(() => {
      this.setupCanvas();
      const rect = this.container.getBoundingClientRect();
      this.ctx.clearRect(0, 0, rect.width, rect.height);
      this.drawCurve();
      this.drawPoints();
    });
  }
  drawCurve() {
    const lut = this.sampleCurve(256);
    this.ctx.beginPath();
    let p = this.toScreen({
      x: 0,
      y: lut[0]
    });
    this.ctx.moveTo(p.x, p.y);
    for (let i = 1; i < lut.length; i++) {
      p = this.toScreen({
        x: i / (lut.length - 1),
        y: lut[i]
      });
      this.ctx.lineTo(p.x, p.y);
    }
    this.ctx.strokeStyle = "blue";
    this.ctx.lineWidth = 2;
    this.ctx.stroke();
  }
  drawPoints() {
    for (let i = 0; i < this.points.length; i++) {
      const p = this.points[i];
      const screenP = this.toScreen(p);
      this.ctx.beginPath();
      this.ctx.arc(screenP.x, screenP.y, POINT_RADIUS + 2, 0, 2 * Math.PI);
      this.ctx.fillStyle = "black";
      this.ctx.fill();
      this.ctx.beginPath();
      this.ctx.arc(screenP.x, screenP.y, POINT_RADIUS, 0, 2 * Math.PI);
      this.ctx.fillStyle = "red";
      this.ctx.fill();
    }
  }
};

// translations/translation.json
var translation_default = '{\n  "en": {\n    "TR_EXPOSURE": "Exposure",\n    "TR_CONTRAST": "Contrast",\n    "TR_SHADOW": "Shadow",\n    "TR_HIGHLIGHT": "Highlight",\n    "TR_BLACK_LEVEL": "Black Level",\n    "TR_WHITE_LEVEL": "White Level",\n    "TR_TEMPERATURE": "Temperature",\n    "TR_TINT": "Tint",\n    "TR_VIGNETTE": "Vignette",\n    "TR_LENS_DISTORTION": "Lens Distortion",\n    "TR_SAVED_FILE": "Saved file: ",\n    "TR_JPEG_IMAGE": "JPEG Image",\n    "TR_PNG_IMAGE": "PNG Image",\n    "TR_MENU_OPEN": "Open Photo",\n    "TR_MENU_SAVE": "Save Photo",\n    "TR_MENU_RESET_ALL": "Reset All Edits",\n    "TR_MENU_SETTING": "Settings",\n    "TR_TAB_TONE": "Tone",\n    "TR_TAB_BRIGHTNESS": "Brightness",\n    "TR_TAB_HUE": "Hue",\n    "TR_TAB_SATURATION": "Saturation",\n    "TR_TAB_LIGHTNESS": "Lightness",\n    "TR_TAB_WB": "White Balance",\n    "TR_TAB_EFFECT": "Effect",\n    "TR_TAB_METADATA": "Metadata",\n    "TR_RESET_THIS_TAB": "Reset This Tab",\n    "TR_DIALOG_OPEN_TITLE": "Open a File",\n    "TR_DIALOG_SAVE_TITLE": "Save Photo",\n    "TR_SELECT_FORMAT": "Select Format",\n    "TR_BUTTON_SAVE": "Save",\n    "TR_BUTTON_CANCEL": "Cancel",\n    "TR_DIALOG_SAVE_COMPLETE_TITLE": "Save Complete",\n    "TR_BUTTON_OK": "OK",\n    "TR_MENU_FILE": "File",\n    "TR_MENU_EDIT": "Edit",\n    "TR_SETTINGS_TITLE": "Settings",\n    "TR_SETTINGS_TAB_LANGUAGE": "Language",\n    "TR_SETTINGS_TAB_IMAGE": "Image",\n    "TR_SETTINGS_UI_PREVIEW_SIZE": "UI Preview Size (Long Edge)",\n    "TR_SETTINGS_DRAG_PREVIEW_SIZE": "Drag Preview Size (Long Edge)",\n    "TR_SETTINGS_SAVE_BUTTON": "Save Settings",\n    "TR_INFO_DIALOG_TITLE": "Information",\n    "TR_SETTINGS_LANGUAGE_LABEL": "Language",\n    "TR_DEVICE": "Device",\n    "TR_ERROR_IMAGE_LOAD": "Failed to load image",\n    "TR_ERROR_IMAGE_SAVE": "Failed to save image",\n    "TR_ERROR_GET_WEBGPU_CANVAS_CONTEXT": "Failed to obtain WebGPU Canvas Context.",\n    "TR_ERROR_WEBGPU": "WebGPU initialization failed.",\n    "TR_SAVE_SUPPORT_IMAGE": "Support image",\n    "TR_SAVE_STANDARD_IMAGE": "Standard image",\n    "TR_SAVE_RAW_IMAGE": "RAW image",\n    "TR_SETTINGS_SAVED_INFO": "Settings have been saved."\n  },\n  "ja": {\n    "TR_EXPOSURE": "\u9732\u51FA",\n    "TR_CONTRAST": "\u30B3\u30F3\u30C8\u30E9\u30B9\u30C8",\n    "TR_SHADOW": "\u30B7\u30E3\u30C9\u30A6",\n    "TR_HIGHLIGHT": "\u30CF\u30A4\u30E9\u30A4\u30C8",\n    "TR_BLACK_LEVEL": "\u9ED2\u30EC\u30D9\u30EB",\n    "TR_WHITE_LEVEL": "\u767D\u30EC\u30D9\u30EB",\n    "TR_TEMPERATURE": "\u8272\u6E29\u5EA6",\n    "TR_TINT": "\u8272\u304B\u3076\u308A\u88DC\u6B63",\n    "TR_VIGNETTE": "\u5468\u8FBA\u6E1B\u5149",\n    "TR_LENS_DISTORTION": "\u30EC\u30F3\u30BA\u6B6A\u307F\u88DC\u6B63",\n    "TR_SAVED_FILE": "\u4FDD\u5B58\u3057\u307E\u3057\u305F: ",\n    "TR_JPEG_IMAGE": "JPEG\u753B\u50CF",\n    "TR_PNG_IMAGE": "PNG\u753B\u50CF",\n    "TR_MENU_OPEN": "\u5199\u771F\u3092\u958B\u304F",\n    "TR_MENU_SAVE": "\u5199\u771F\u3092\u4FDD\u5B58",\n    "TR_MENU_RESET_ALL": "\u3059\u3079\u3066\u306E\u7DE8\u96C6\u3092\u30EA\u30BB\u30C3\u30C8",\n    "TR_MENU_SETTING": "\u8A2D\u5B9A",\n    "TR_TAB_TONE": "\u30C8\u30FC\u30F3",\n    "TR_TAB_BRIGHTNESS": "\u660E\u308B\u3055",\n    "TR_TAB_HUE": "\u8272\u76F8",\n    "TR_TAB_SATURATION": "\u5F69\u5EA6",\n    "TR_TAB_LIGHTNESS": "\u8F1D\u5EA6",\n    "TR_TAB_WB": "\u30DB\u30EF\u30A4\u30C8\u30D0\u30E9\u30F3\u30B9",\n    "TR_TAB_EFFECT": "\u52B9\u679C",\n    "TR_TAB_METADATA": "\u30E1\u30BF\u30C7\u30FC\u30BF",\n    "TR_RESET_THIS_TAB": "\u3053\u306E\u30BF\u30D6\u3092\u30EA\u30BB\u30C3\u30C8",\n    "TR_DIALOG_OPEN_TITLE": "\u30D5\u30A1\u30A4\u30EB\u3092\u958B\u304F",\n    "TR_DIALOG_SAVE_TITLE": "\u5199\u771F\u3092\u4FDD\u5B58",\n    "TR_SELECT_FORMAT": "\u30D5\u30A9\u30FC\u30DE\u30C3\u30C8\u3092\u9078\u629E",\n    "TR_BUTTON_SAVE": "\u4FDD\u5B58",\n    "TR_BUTTON_CANCEL": "\u30AD\u30E3\u30F3\u30BB\u30EB",\n    "TR_DIALOG_SAVE_COMPLETE_TITLE": "\u4FDD\u5B58\u5B8C\u4E86",\n    "TR_BUTTON_OK": "OK",\n    "TR_MENU_FILE": "\u30D5\u30A1\u30A4\u30EB",\n    "TR_MENU_EDIT": "\u7DE8\u96C6",\n    "TR_SETTINGS_TITLE": "\u8A2D\u5B9A",\n    "TR_SETTINGS_TAB_LANGUAGE": "\u8A00\u8A9E",\n    "TR_SETTINGS_TAB_IMAGE": "\u753B\u50CF",\n    "TR_SETTINGS_UI_PREVIEW_SIZE": "UI\u30D7\u30EC\u30D3\u30E5\u30FC\u30B5\u30A4\u30BA (\u9577\u8FBA)",\n    "TR_SETTINGS_DRAG_PREVIEW_SIZE": "\u30C9\u30E9\u30C3\u30B0\u4E2D\u30D7\u30EC\u30D3\u30E5\u30FC\u30B5\u30A4\u30BA (\u9577\u8FBA)",\n    "TR_SETTINGS_SAVE_BUTTON": "\u8A2D\u5B9A\u3092\u4FDD\u5B58",\n    "TR_INFO_DIALOG_TITLE": "\u60C5\u5831",\n    "TR_SETTINGS_LANGUAGE_LABEL": "\u8A00\u8A9E",\n    "TR_DEVICE": "\u30C7\u30D0\u30A4\u30B9",\n    "TR_ERROR_IMAGE_LOAD": "\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ",\n    "TR_ERROR_IMAGE_SAVE": "\u753B\u50CF\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",\n    "TR_ERROR_GET_WEBGPU_CANVAS_CONTEXT": "WebGPU Canvas Context\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",\n    "TR_ERROR_WEBGPU": "WebGPU\u306E\u521D\u671F\u5316\u306B\u5931\u6557\u3057\u307E\u3057\u305F",\n    "TR_SAVE_SUPPORT_IMAGE": "\u5BFE\u5FDC\u753B\u50CF",\n    "TR_SAVE_STANDARD_IMAGE": "\u6A19\u6E96\u753B\u50CF",\n    "TR_SAVE_RAW_IMAGE": "RAW\u753B\u50CF",\n    "TR_SETTINGS_SAVED_INFO": "\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F\u3002"\n  }\n}';

// main.ts
var defaultSettings = {
  uiPreviewSize: 1280,
  dragPreviewSize: 400,
  locale: "en"
};
var settings = {
  ...defaultSettings
};
var SETTINGS_FILE_PATH = "raw-photo-forge-settings";
var I18n = class {
  lang;
  data;
  constructor(data2, lang) {
    this.data = data2;
    this.lang = lang;
    console.log(data2);
  }
  t(key) {
    return this.data[this.lang]?.[key] ?? this.data["en"]?.[key] ?? key;
  }
  setLang(lang) {
    this.lang = lang;
  }
};
var PreviewLevel = /* @__PURE__ */ function(PreviewLevel2) {
  PreviewLevel2[PreviewLevel2["LOW"] = 0] = "LOW";
  PreviewLevel2[PreviewLevel2["MID"] = 1] = "MID";
  PreviewLevel2[PreviewLevel2["FULL"] = 2] = "FULL";
  return PreviewLevel2;
}(PreviewLevel || {});
var gpuProcessor;
var editorFull = null;
var editorMid = null;
var editorLow = null;
var currentImageFile = null;
var imageLoaded = false;
var previewLevel = PreviewLevel.MID;
var uniformBuffer;
var toneCurveEditors = {};
var initialEditState = {
  exposure: 0,
  contrast: 0,
  shadow: 0,
  highlight: 0,
  black: 0,
  white: 0,
  temperature: 0,
  tint: 0,
  vignette: 0,
  lens_distortion: 0,
  brightness_tone_curve_points: [
    {
      x: 0,
      y: 0
    },
    {
      x: 1,
      y: 1
    }
  ],
  hue_tone_curve_points: [
    {
      x: 0,
      y: 0
    },
    {
      x: 1,
      y: 1
    }
  ],
  saturation_tone_curve_points: [
    {
      x: 0,
      y: 1
    },
    {
      x: 1,
      y: 1
    }
  ],
  lightness_tone_curve_points: [
    {
      x: 0,
      y: 1
    },
    {
      x: 1,
      y: 1
    }
  ]
};
var editState = {
  ...initialEditState
};
var canvasContext = null;
var presentationFormat;
var renderPipeline = null;
var isRendering = false;
var data = JSON.parse(translation_default);
console.log(data);
var i18n = new I18n(data, "en");
var renderShaderCode = `
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
var ui = {
  mainCanvas: document.getElementById("main-canvas"),
  fileInput: document.getElementById("file-input"),
  exposureLabel: document.getElementById("exposure-label"),
  contrastLabel: document.getElementById("contrast-label"),
  shadowLabel: document.getElementById("shadow-label"),
  highlightLabel: document.getElementById("highlight-label"),
  blackLabel: document.getElementById("black-label"),
  whiteLabel: document.getElementById("white-label"),
  temperatureLabel: document.getElementById("temperature-label"),
  tintLabel: document.getElementById("tint-label"),
  vignetteLabel: document.getElementById("vignette-label"),
  lensDistortionLabel: document.getElementById("lens-distortion-label"),
  exposureSlider: document.getElementById("exposure-slider"),
  contrastSlider: document.getElementById("contrast-slider"),
  shadowSlider: document.getElementById("shadow-slider"),
  highlightSlider: document.getElementById("highlight-slider"),
  blackSlider: document.getElementById("black-slider"),
  whiteSlider: document.getElementById("white-slider"),
  temperatureSlider: document.getElementById("temperature-slider"),
  tintSlider: document.getElementById("tint-slider"),
  vignetteSlider: document.getElementById("vignette-slider"),
  lensDistortionSlider: document.getElementById("lens-distortion-slider"),
  resetToneButton: document.getElementById("reset-tone-button"),
  resetWbButton: document.getElementById("reset-wb-button"),
  resetEffectButton: document.getElementById("reset-effect-button"),
  resetBrightnessButton: document.getElementById("reset-brightness-button"),
  resetHueButton: document.getElementById("reset-hue-button"),
  resetSaturationButton: document.getElementById("reset-saturation-button"),
  resetLightnessButton: document.getElementById("reset-lightness-button"),
  tabButtons: document.querySelectorAll(".tab-button"),
  tabPanes: document.querySelectorAll(".tab-pane"),
  openFile: document.getElementById("open-file"),
  saveFile: document.getElementById("save-file"),
  resetAll: document.getElementById("reset-all"),
  saveDialog: document.getElementById("save-dialog"),
  saveDialogSave: document.getElementById("save-dialog-save"),
  saveDialogCancel: document.getElementById("save-dialog-cancel"),
  formatSelect: document.getElementById("format-select"),
  settingsMenu: document.getElementById("settings-menu"),
  settingsDialog: document.getElementById("settings-dialog"),
  settingsDialogSave: document.getElementById("settings-dialog-save"),
  settingsDialogCancel: document.getElementById("settings-dialog-cancel"),
  uiPreviewSizeSlider: document.getElementById("ui-preview-size-slider"),
  uiPreviewSizeInput: document.getElementById("ui-preview-size-input"),
  dragPreviewSizeSlider: document.getElementById("drag-preview-size-slider"),
  dragPreviewSizeInput: document.getElementById("drag-preview-size-input"),
  languageSelect: document.getElementById("language-select"),
  infoDialog: document.getElementById("info-dialog"),
  infoDialogText: document.getElementById("info-dialog-text"),
  infoDialogOk: document.getElementById("info-dialog-ok"),
  metadataTree: document.getElementById("metadata-tree")
};
function applyI18n(i18n2) {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = i18n2.t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.placeholder = i18n2.t(el.dataset.i18nPlaceholder);
  });
  document.querySelectorAll("[data-i18n-alt]").forEach((el) => {
    el.alt = i18n2.t(el.dataset.i18nAlt);
  });
}
function showInfoDialog(message) {
  ui.infoDialogText.textContent = message;
  ui.infoDialog.style.display = "flex";
}
function loadSettings() {
  const savedSettings = localStorage.getItem(SETTINGS_FILE_PATH);
  if (savedSettings) {
    try {
      const parsed = JSON.parse(savedSettings);
      settings = {
        ...defaultSettings,
        ...parsed
      };
    } catch (e) {
      console.error("Failed to parse settings, using defaults.", e);
      settings = {
        ...defaultSettings
      };
    }
  } else {
    const browserLang = navigator.language.split("-")[0];
    if (browserLang === "ja") {
      defaultSettings.locale = "ja";
    }
    settings = {
      ...defaultSettings
    };
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
      m.addedNodes.forEach((node) => {
        if (node instanceof HTMLElement) {
          if (node.dataset.i18n) {
            node.textContent = i18n.t(node.dataset.i18n);
          }
          node.querySelectorAll?.("[data-i18n]").forEach((el) => {
            el.textContent = i18n.t(el.dataset.i18n);
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
    console.log("WebGPU initialized successfully.");
  } catch (error) {
    console.error("WebGPU initialization failed:", error);
    alert(`${i18n.t("TR_ERROR_WEBGPU")}\u3002${error}`);
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
    alphaMode: "premultiplied"
  });
  const device = gpuProcessor.getDevice();
  const shaderModule = device.createShaderModule({
    label: "Render Shader Module",
    code: renderShaderCode
  });
  const renderBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {
          sampleType: "unfilterable-float"
        }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: {
          type: "uniform"
        }
      }
    ]
  });
  uniformBuffer = device.createBuffer({
    size: 4 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  renderPipeline = device.createRenderPipeline({
    label: "Render to Canvas Pipeline",
    layout: device.createPipelineLayout({
      bindGroupLayouts: [
        renderBindGroupLayout
      ]
    }),
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main"
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [
        {
          format: presentationFormat
        }
      ]
    },
    primitive: {
      topology: "triangle-strip",
      stripIndexFormat: "uint32"
    }
  });
}
function setupEventListeners() {
  ui.openFile.addEventListener("click", () => ui.fileInput.click());
  ui.fileInput.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) {
      currentImageFile = file;
      loadImage(file);
    }
  });
  ui.saveFile.addEventListener("click", () => {
    if (!imageLoaded) return;
    ui.saveDialog.style.display = "flex";
  });
  ui.resetAll.addEventListener("click", resetAllEdits);
  ui.saveDialogCancel.addEventListener("click", () => ui.saveDialog.style.display = "none");
  ui.saveDialogSave.addEventListener("click", saveImage);
  ui.tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const tabName = button.dataset.tab;
      ui.tabButtons.forEach((btn) => btn.classList.remove("active"));
      ui.tabPanes.forEach((pane) => pane.classList.remove("active"));
      button.classList.add("active");
      const newPane = document.getElementById(`tab-${tabName}`);
      if (newPane) {
        newPane.classList.add("active");
      }
      if (toneCurveEditors[tabName]) {
        toneCurveEditors[tabName].draw();
      }
    });
  });
  const sliders = [
    {
      s: ui.exposureSlider,
      k: "exposure",
      l: ui.exposureLabel,
      n: i18n.t("TR_EXPOSURE"),
      f: (v) => v.toFixed(2)
    },
    {
      s: ui.contrastSlider,
      k: "contrast",
      l: ui.contrastLabel,
      n: i18n.t("TR_CONTRAST"),
      f: (v) => Math.round(v)
    },
    {
      s: ui.shadowSlider,
      k: "shadow",
      l: ui.shadowLabel,
      n: i18n.t("TR_SHADOW"),
      f: (v) => Math.round(v)
    },
    {
      s: ui.highlightSlider,
      k: "highlight",
      l: ui.highlightLabel,
      n: i18n.t("TR_HIGHLIGHT"),
      f: (v) => Math.round(v)
    },
    {
      s: ui.blackSlider,
      k: "black",
      l: ui.blackLabel,
      n: i18n.t("TR_BLACK_LEVEL"),
      f: (v) => Math.round(v)
    },
    {
      s: ui.whiteSlider,
      k: "white",
      l: ui.whiteLabel,
      n: i18n.t("TR_WHITE_LEVEL"),
      f: (v) => Math.round(v)
    },
    {
      s: ui.temperatureSlider,
      k: "temperature",
      l: ui.temperatureLabel,
      n: i18n.t("TR_TEMPERATURE"),
      f: (v) => Math.round(v)
    },
    {
      s: ui.tintSlider,
      k: "tint",
      l: ui.tintLabel,
      n: i18n.t("TR_TINT"),
      f: (v) => Math.round(v)
    },
    {
      s: ui.vignetteSlider,
      k: "vignette",
      l: ui.vignetteLabel,
      n: i18n.t("TR_VIGNETTE"),
      f: (v) => Math.round(v)
    },
    {
      s: ui.lensDistortionSlider,
      k: "lens_distortion",
      l: ui.lensDistortionLabel,
      n: i18n.t("TR_LENS_DISTORTION"),
      f: (v) => Math.round(v)
    }
  ];
  sliders.forEach(({ s, k, l, n, f }) => {
    s.addEventListener("input", () => {
      const value = parseFloat(s.value);
      editState[k] = value;
      l.textContent = `${n} ${f(value)}`;
      updateImage();
    });
    s.addEventListener("mousedown", onDragStart);
    s.addEventListener("mouseup", onDragEnd);
  });
  ui.resetToneButton.addEventListener("click", resetTone);
  ui.resetWbButton.addEventListener("click", resetWb);
  ui.resetEffectButton.addEventListener("click", resetEffect);
  ui.resetBrightnessButton.addEventListener("click", () => resetCurve("brightness"));
  ui.resetHueButton.addEventListener("click", () => resetCurve("hue"));
  ui.resetSaturationButton.addEventListener("click", () => resetCurve("saturation"));
  ui.resetLightnessButton.addEventListener("click", () => resetCurve("lightness"));
  ui.settingsMenu.addEventListener("click", () => {
    updateSettingsUI();
    ui.settingsDialog.style.display = "flex";
  });
  ui.settingsDialogCancel.addEventListener("click", () => {
    ui.settingsDialog.style.display = "none";
  });
  ui.settingsDialogSave.addEventListener("click", () => {
    if (saveSettings()) {
      applySettings();
      ui.settingsDialog.style.display = "none";
      showInfoDialog(i18n.t("TR_SETTINGS_SAVED_INFO"));
      if (currentImageFile) {
        loadImage(currentImageFile);
      }
    }
  });
  ui.infoDialogOk.addEventListener("click", () => {
    ui.infoDialog.style.display = "none";
  });
  ui.uiPreviewSizeSlider.addEventListener("input", () => {
    ui.uiPreviewSizeInput.value = ui.uiPreviewSizeSlider.value;
  });
  ui.uiPreviewSizeInput.addEventListener("change", () => {
    let value = parseInt(ui.uiPreviewSizeInput.value, 10);
    const min = parseInt(ui.uiPreviewSizeSlider.min, 10);
    const max = parseInt(ui.uiPreviewSizeSlider.max, 10);
    if (isNaN(value) || value < min) value = min;
    if (value > max) value = max;
    ui.uiPreviewSizeInput.value = String(value);
    ui.uiPreviewSizeSlider.value = String(value);
  });
  ui.dragPreviewSizeSlider.addEventListener("input", () => {
    ui.dragPreviewSizeInput.value = ui.dragPreviewSizeSlider.value;
  });
  ui.dragPreviewSizeInput.addEventListener("change", () => {
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
  const onCurveChange = (key) => (points) => {
    editState[key] = points;
    updateImage();
  };
  toneCurveEditors["brightness"] = new ToneCurveEditor("brightness-tone-curve-editor", CurveMode.BRIGHTNESS, onCurveChange("brightness_tone_curve_points"), onDragStart, onDragEnd);
  toneCurveEditors["hue"] = new ToneCurveEditor("hue-tone-curve-editor", CurveMode.HUE, onCurveChange("hue_tone_curve_points"), onDragStart, onDragEnd);
  toneCurveEditors["saturation"] = new ToneCurveEditor("saturation-tone-curve-editor", CurveMode.SATURATION, onCurveChange("saturation_tone_curve_points"), onDragStart, onDragEnd);
  toneCurveEditors["lightness"] = new ToneCurveEditor("lightness-tone-curve-editor", CurveMode.LIGHTNESS, onCurveChange("lightness_tone_curve_points"), onDragStart, onDragEnd);
  toneCurveEditors["brightness"].setBackground("./assets/tone_curve/brightness_gradient.png");
  toneCurveEditors["hue"].setBackground("./assets/tone_curve/hue_bars.png");
  toneCurveEditors["saturation"].setBackground("./assets/tone_curve/hue_vs_saturation.png");
  toneCurveEditors["lightness"].setBackground("./assets/tone_curve/hue_vs_lightness.png");
  Object.values(toneCurveEditors).forEach((editor) => {
    editState.brightness_tone_curve_points = editor.points;
  });
}
async function loadImage(file) {
  let originalImageBitmap;
  let float32ArrayImage;
  const midResLongEdge = settings.uiPreviewSize;
  const lowResLongEdge = settings.dragPreviewSize;
  if (file.name.toLowerCase().endsWith(".ppm")) {
    float32ArrayImage = await loadPpm(file);
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
async function renderProcessedTextureToCanvas(textureView, width, height) {
  if (!canvasContext || !renderPipeline) return;
  const device = gpuProcessor.getDevice();
  const queue = gpuProcessor.getQueue();
  const data2 = new Float32Array([
    ui.mainCanvas.width,
    ui.mainCanvas.height,
    width,
    height
  ]);
  queue.writeBuffer(uniformBuffer, 0, data2.buffer);
  const bindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: textureView
      },
      {
        binding: 1,
        resource: {
          buffer: uniformBuffer
        }
      }
    ]
  });
  const canvasTexture = canvasContext.getCurrentTexture();
  const commandEncoder = device.createCommandEncoder();
  const pass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: canvasTexture.createView(),
        clearValue: {
          r: 0,
          g: 0,
          b: 0,
          a: 1
        },
        loadOp: "clear",
        storeOp: "store"
      }
    ]
  });
  pass.setPipeline(renderPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(4);
  pass.end();
  queue.submit([
    commandEncoder.finish()
  ]);
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
  let editor;
  switch (previewLevel) {
    case PreviewLevel.LOW:
      editor = editorLow;
      break;
    case PreviewLevel.MID:
      editor = editorMid;
      break;
    case PreviewLevel.FULL:
      editor = editorFull;
      break;
  }
  setEditorParameters(editor);
  await editor.applyAdjustments();
  console.log("applyAdjustments", Date.now() - s);
  await renderProcessedTextureToCanvas(editor.image.textureView, editor.image.width, editor.image.height);
  isRendering = false;
  console.log("applyAdjustments+render", Date.now() - s);
}
function setEditorParameters(e) {
  const s = editState;
  e.setTone(s.exposure, s.contrast, s.shadow, s.highlight, s.black, s.white);
  e.setWhitebalance(s.temperature, s.tint);
  e.setVignette(s.vignette);
  e.setLensDistortionCorrection(s.lens_distortion);
  const toPoints = (points) => points.map((p) => p.x * 65535);
  const toValues = (points) => points.map((p) => p.y * 65535);
  e.setBrightnessToneCurve(void 0, toPoints(s.brightness_tone_curve_points), toValues(s.brightness_tone_curve_points));
  e.setOklchHueCurve(void 0, toPoints(s.hue_tone_curve_points), toValues(s.hue_tone_curve_points));
  const toSatLightValues = (points) => points.map((p) => p.y / 2 * 65535);
  e.setOklchSaturationCurve(void 0, toPoints(s.saturation_tone_curve_points), toSatLightValues(s.saturation_tone_curve_points));
  e.setOklchLightnessCurve(void 0, toPoints(s.lightness_tone_curve_points), toSatLightValues(s.lightness_tone_curve_points));
}
function resetAllEdits() {
  resetCurve("brightness");
  resetCurve("hue");
  resetCurve("saturation");
  resetCurve("lightness");
  resetTone();
  resetWb();
  resetEffect();
}
function resetTone() {
  editState.exposure = 0;
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
function resetCurve(name) {
  if (toneCurveEditors[name]) {
    toneCurveEditors[name].initializePoints();
    if (name === "brightness" || name === "hue") {
      toneCurveEditors[name].points = [
        {
          x: 0,
          y: 0
        },
        {
          x: 1,
          y: 1
        }
      ];
    } else {
      toneCurveEditors[name].points = [
        {
          x: 0,
          y: 1
        },
        {
          x: 1,
          y: 1
        }
      ];
    }
    editState[`${name}_tone_curve_points`] = toneCurveEditors[name].points;
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
  ui.saveDialog.style.display = "none";
  previewLevel = PreviewLevel.FULL;
  setEditorParameters(editorFull);
  await editorFull.applyAdjustments();
  const finalBitmap = await editorFull.save();
  const canvas = document.createElement("canvas");
  canvas.width = finalBitmap.width;
  canvas.height = finalBitmap.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(finalBitmap, 0, 0);
  const format = ui.formatSelect.value === "jpeg" ? "image/jpeg" : "image/png";
  const quality = format === "image/jpeg" ? 0.9 : void 0;
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, format, quality));
  if (!blob) {
    alert(i18n.t("TR_ERROR_IMAGE_SAVE"));
    return;
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const basename = currentImageFile.name.split(".").slice(0, -1).join(".");
  const ext = ui.formatSelect.value;
  a.href = url;
  a.download = `${basename}_edited.${ext}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
initializeApp();
