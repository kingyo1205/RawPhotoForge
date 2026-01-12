// wgpu_shader.wgsl

//--------------------------------------------------------------------------------
// Resources
//--------------------------------------------------------------------------------

@group(0) @binding(0)
var<storage, read_write> image: array<f32>;

@group(0) @binding(2)
var<storage, read> curve: array<f32>;

@group(0) @binding(3)
var<storage, read> mask: array<u32>;

@group(0) @binding(4)
var<storage, read> channels: array<u32>;

//--------------------------------------------------------------------------------
// Uniforms
//--------------------------------------------------------------------------------

struct BaseParams {
    width: u32,
    height: u32,
};
@group(0) @binding(1)
var<uniform> base_params: BaseParams;

struct ToneCurveParams {
    width: u32,
    height: u32,
    channel_count: u32,
};
@group(0) @binding(5)
var<uniform> tone_curve_params: ToneCurveParams;

struct ToneCurveByHueParams {
    width: u32,
    height: u32,
    ch_hue: u32,
    ch_target: u32,
};
@group(0) @binding(6)
var<uniform> tone_curve_by_hue_params: ToneCurveByHueParams;

struct WhiteBalanceParams {
    width: u32,
    height: u32,
    r_gain: f32,
    g_gain: f32,
    b_gain: f32,
};
@group(0) @binding(7)
var<uniform> wb_params: WhiteBalanceParams;

struct VignetteParams {
    width: u32,
    height: u32,
    strength: f32,
};
@group(0) @binding(8)
var<uniform> vignette_params: VignetteParams;

//--------------------------------------------------------------------------------
// OKLCH Conversion Matrices and Functions
//--------------------------------------------------------------------------------

const M1: mat3x3<f32> = mat3x3<f32>(
    0.4121656120, 0.2118591070, 0.0883097947,
    0.5362752080, 0.6807189584, 0.2818474174,
    0.0514575653, 0.1074065553, 0.6302613616
);

const M1_INV: mat3x3<f32> = mat3x3<f32>(
    4.0767245293, -1.2681437731, -0.0041119885,
   -3.3072168827,  2.6093323231, -0.7034763098,
    0.2307590544, -0.3411344290,  1.7068625689
);

const M2: mat3x3<f32> = mat3x3<f32>(
     0.2104542553,  1.9779984951,  0.0259040371,
     0.7936177850, -2.4285922050,  0.7827717662,
    -0.0040720468,  0.4505937099, -0.8086757660
);

const M2_INV: mat3x3<f32> = mat3x3<f32>(
    1.0,  1.0,  1.0,
    0.3963377774, -0.1055613458, -0.0894841775,
    0.2158037573, -0.0638541728, -1.2914855480
);

fn linear_srgb_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let lms = M1 * c;
    let lms_cbrt = pow(lms, vec3<f32>(1.0/3.0));
    return M2 * lms_cbrt;
}

fn oklab_to_linear_srgb(c: vec3<f32>) -> vec3<f32> {
    let lms_cbrt = M2_INV * c;
    let lms = pow(lms_cbrt, vec3<f32>(3.0));
    return M1_INV * lms;
}

fn oklab_to_oklch(c: vec3<f32>) -> vec3<f32> {
    let L = c.x;
    let a = c.y;
    let b = c.z;
    let C = sqrt(a * a + b * b);
    var h = atan2(b, a) / (2.0 * 3.14159265359);
    if (h < 0.0) {
        h = h + 1.0;
    }
    return vec3<f32>(L, C, h);
}

fn oklch_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let L = c.x;
    let C = c.y;
    let h = c.z * 2.0 * 3.14159265359;
    let a = C * cos(h);
    let b = C * sin(h);
    return vec3<f32>(L, a, b);
}

//--------------------------------------------------------------------------------
// Compute Kernels
//--------------------------------------------------------------------------------

@compute
@workgroup_size(16, 16, 1)
fn to_linear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= base_params.width || global_id.y >= base_params.height) {
        return;
    }
    let pixel_idx = global_id.y * base_params.width + global_id.x;
    let base_idx = pixel_idx * 3u;

    for (var i: u32 = 0; i < 3; i = i + 1) {
        let x = clamp(image[base_idx + i], 0.0, 1.0);
        var result: f32;
        if (x <= 0.04045) {
            result = x / 12.92;
        } else {
            result = pow((x + 0.055) / 1.055, 2.4);
        }
        image[base_idx + i] = clamp(result, 0.0, 1.0);
    }
}

@compute
@workgroup_size(16, 16, 1)
fn clip_0_1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= base_params.width || global_id.y >= base_params.height) {
        return;
    }
    let pixel_idx = global_id.y * base_params.width + global_id.x;
    let base_idx = pixel_idx * 3u;

    for (var i: u32 = 0; i < 3; i = i + 1) {
        image[base_idx + i] = clamp(image[base_idx + i], 0.0, 1.0);
    }
}

@compute
@workgroup_size(16, 16, 1)
fn to_srgb(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= base_params.width || global_id.y >= base_params.height) {
        return;
    }
    let pixel_idx = global_id.y * base_params.width + global_id.x;
    let base_idx = pixel_idx * 3u;

    for (var i: u32 = 0; i < 3; i = i + 1) {
        let x = clamp(image[base_idx + i], 0.0, 1.0);
        var result: f32;
        if (x <= 0.0031308) {
            result = x * 12.92;
        } else {
            result = 1.055 * pow(x, 1.0 / 2.4) - 0.055;
        }
        image[base_idx + i] = clamp(result, 0.0, 1.0);
    }
}

@compute
@workgroup_size(16, 16, 1)
fn tone_curve_lut(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= tone_curve_params.width || global_id.y >= tone_curve_params.height) {
        return;
    }
    let pixel_idx = global_id.y * tone_curve_params.width + global_id.x;

    if (mask[pixel_idx] != 0u) {
        let base_idx = pixel_idx * 3;
        for (var ch: u32 = 0; ch < tone_curve_params.channel_count; ch = ch + 1) {
            let c = channels[ch];
            let val = clamp(image[base_idx + c], 0.0, 1.0);
            var lut_idx = u32(val * 65535.0);
            lut_idx = clamp(lut_idx, 0u, 65535u);
            let result = curve[lut_idx] / 65535.0;
            image[base_idx + c] = clamp(result, 0.0, 1.0);
        }
    }
}

@compute
@workgroup_size(16, 16, 1)
fn tone_curve_by_hue(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= tone_curve_by_hue_params.width || global_id.y >= tone_curve_by_hue_params.height) {
        return;
    }
    let pixel_idx = global_id.y * tone_curve_by_hue_params.width + global_id.x;

    if (mask[pixel_idx] != 0u) {
        let base_idx = pixel_idx * 3;
        let hue = clamp(image[base_idx + tone_curve_by_hue_params.ch_hue], 0.0, 1.0);
        let val = clamp(image[base_idx + tone_curve_by_hue_params.ch_target], 0.0, 1.0);

        var hue_idx = u32(hue * 65535.0);
        hue_idx = clamp(hue_idx, 0u, 65535u);

        let gain = curve[hue_idx] / 65535.0;
        let new_val = clamp(val * gain, 0.0, 1.0);

        image[base_idx + tone_curve_by_hue_params.ch_target] = new_val;
    }
}

@compute
@workgroup_size(16, 16, 1)
fn rgb_to_oklch(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= base_params.width || global_id.y >= base_params.height) {
        return;
    }
    let pixel_idx = global_id.y * base_params.width + global_id.x;
    let base_idx = pixel_idx * 3u;

    let rgb_linear = vec3<f32>(image[base_idx], image[base_idx + 1u], image[base_idx + 2u]);
    let oklab = linear_srgb_to_oklab(rgb_linear);
    let oklch = oklab_to_oklch(oklab);

    image[base_idx] = oklch.x;
    image[base_idx + 1u] = oklch.y;
    image[base_idx + 2u] = oklch.z;
}

@compute
@workgroup_size(16, 16, 1)
fn oklch_to_rgb(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= base_params.width || global_id.y >= base_params.height) {
        return;
    }
    let pixel_idx = global_id.y * base_params.width + global_id.x;
    let base_idx = pixel_idx * 3u;

    let oklch = vec3<f32>(image[base_idx], image[base_idx + 1u], image[base_idx + 2u]);
    let oklab = oklch_to_oklab(oklch);
    let rgb_linear = oklab_to_linear_srgb(oklab);

    image[base_idx] = rgb_linear.x;
    image[base_idx + 1u] = rgb_linear.y;
    image[base_idx + 2u] = rgb_linear.z;
}

@compute
@workgroup_size(16, 16, 1)
fn white_balance(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= wb_params.width || global_id.y >= wb_params.height) {
        return;
    }
    let pixel_idx = global_id.y * wb_params.width + global_id.x;

    if (mask[pixel_idx] != 0u) {
        let base_idx = pixel_idx * 3;
        var r = clamp(image[base_idx + 0], 0.0, 1.0);
        var g = clamp(image[base_idx + 1], 0.0, 1.0);
        var b = clamp(image[base_idx + 2], 0.0, 1.0);

        r = r * wb_params.r_gain;
        g = g * wb_params.g_gain;
        b = b * wb_params.b_gain;

        image[base_idx + 0] = clamp(r, 0.0, 1.0);
        image[base_idx + 1] = clamp(g, 0.0, 1.0);
        image[base_idx + 2] = clamp(b, 0.0, 1.0);
    }
}

@compute
@workgroup_size(16, 16, 1)
fn vignette_effect(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= vignette_params.width || global_id.y >= vignette_params.height) {
        return;
    }
    let pixel_idx = global_id.y * vignette_params.width + global_id.x;
    
    if (mask[pixel_idx] != 0u) {
        let x = global_id.x;
        let y = global_id.y;

        let cx = f32(vignette_params.width) * 0.5;
        let cy = f32(vignette_params.height) * 0.5;
        let dx = (f32(x) - cx) / cx;
        let dy = (f32(y) - cy) / cy;

        let dist = sqrt(dx * dx + dy * dy);

        let inner_radius = 0.25;
        let adjusted_dist = max(0.0, (dist - inner_radius) / (1.0 - inner_radius));

        let t = clamp(adjusted_dist, 0.0, 1.0);
        let smooth_t = t * t * (3.0 - 2.0 * t);
        let falloff = pow(smooth_t, 4.0);

        var vignette = 1.0 - (vignette_params.strength * falloff);
        vignette = clamp(vignette, 0.0, 2.0);

        let base_idx = pixel_idx * 3;
        image[base_idx + 0] = clamp(image[base_idx + 0] * vignette, 0.0, 1.0);
        image[base_idx + 1] = clamp(image[base_idx + 1] * vignette, 0.0, 1.0);
        image[base_idx + 2] = clamp(image[base_idx + 2] * vignette, 0.0, 1.0);
    }
}

//--------------------------------------------------------------------------------
// Lens Distortion
//--------------------------------------------------------------------------------

struct LensDistortionParams {
    width: u32,
    height: u32,
    strength: f32,
};
@group(0) @binding(9)
var<uniform> lens_distortion_params: LensDistortionParams;

@group(0) @binding(11)
var<storage, read> image_in: array<f32>;

@group(0) @binding(12)
var<storage, read_write> image_out: array<f32>;


// Bilinear interpolation helper for lens distortion
fn texture_lerp_distortion(uv: vec2<f32>, width: f32, height: f32) -> vec3<f32> {
    let x = uv.x * (width - 1.0);
    let y = uv.y * (height - 1.0);

    let x0 = floor(x);
    let y0 = floor(y);
    let x1 = x0 + 1.0;
    let y1 = y0 + 1.0;

    let fx = fract(x);
    let fy = fract(y);

    let i00_idx = u32(y0 * width + x0) * 3u;
    let i01_idx = u32(y0 * width + x1) * 3u;
    let i10_idx = u32(y1 * width + x0) * 3u;
    let i11_idx = u32(y1 * width + x1) * 3u;
    
    // Bounds checking
    let max_idx = u32(width * height * 3.0) - 3u;
    if (i00_idx > max_idx || i01_idx > max_idx || i10_idx > max_idx || i11_idx > max_idx) {
        // Return black or a border color if out of bounds
        return vec3(0.0, 0.0, 0.0);
    }
    
    let c00 = vec3(image_in[i00_idx], image_in[i00_idx + 1u], image_in[i00_idx + 2u]);
    let c01 = vec3(image_in[i01_idx], image_in[i01_idx + 1u], image_in[i01_idx + 2u]);
    let c10 = vec3(image_in[i10_idx], image_in[i10_idx + 1u], image_in[i10_idx + 2u]);
    let c11 = vec3(image_in[i11_idx], image_in[i11_idx + 1u], image_in[i11_idx + 2u]);

    let top = mix(c00, c01, fx);
    let bottom = mix(c10, c11, fx);
    return mix(top, bottom, fy);
}

@compute
@workgroup_size(16, 16, 1)
fn lens_distortion_effect(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let p = lens_distortion_params;
    if (global_id.x >= p.width || global_id.y >= p.height) {
        return;
    }

    // Normalized texture coordinates (from 0.0 to 1.0) for the output pixel
    let uv = vec2<f32>(f32(global_id.x) / f32(p.width - 1u), f32(global_id.y) / f32(p.height - 1u));
    
    // Convert to centered coordinates [-0.5, 0.5]
    let centered_uv = uv - 0.5;

    // Correct for aspect ratio to make the distortion circular
    let aspect_ratio = f32(p.width) / f32(p.height);
    var corrected_uv = centered_uv;
    corrected_uv.x = corrected_uv.x * aspect_ratio;

    let r2 = dot(corrected_uv, corrected_uv); // radius squared
    
    let k = p.strength;
    
    // Radial distortion formula to find the source coordinate for a given destination coordinate
    // uv_source = uv_dest / (1.0 + k * r_dest^2)
    let distorted_uv_centered = corrected_uv / (1.0 + k * r2);
    
    // Un-correct aspect ratio
    var final_uv_centered = distorted_uv_centered;
    final_uv_centered.x = final_uv_centered.x / aspect_ratio;
    
    // Convert back to [0.0, 1.0] range to sample from the source image
    let final_uv = final_uv_centered + 0.5;
    
    let out_pixel_idx = global_id.y * p.width + global_id.x;
    let out_base_idx = out_pixel_idx * 3u;
    
    var final_color = vec3(0.0, 0.0, 0.0);
    // Sample only if the calculated source coordinates are within the image bounds
    if (final_uv.x >= 0.0 && final_uv.x <= 1.0 && final_uv.y >= 0.0 && final_uv.y <= 1.0) {
        final_color = texture_lerp_distortion(final_uv, f32(p.width), f32(p.height));
    }
    
    image_out[out_base_idx + 0u] = final_color.r;
    image_out[out_base_idx + 1u] = final_color.g;
    image_out[out_base_idx + 2u] = final_color.b;
}