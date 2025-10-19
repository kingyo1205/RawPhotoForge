// wgpu_kernel.wgsl

@group(0) @binding(0)
var<storage, read_write> image: array<f32>;

struct Params {
    num_elements: u32,
    stride_y: u32,
};

@group(0) @binding(1)
var<uniform> params: Params;

@compute
@workgroup_size(256, 1, 1)
fn to_linear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y * params.stride_y + global_id.x;
    if (i >= params.num_elements) {
        return;
    }

    let x = clamp(image[i], 0.0, 1.0);
    var result: f32;
    if (x <= 0.04045) {
        result = x / 12.92;
    } else {
        result = pow((x + 0.055) / 1.055, 2.4);
    }
    image[i] = clamp(result, 0.0, 1.0);
}

@compute
@workgroup_size(256, 1, 1)
fn clip_0_1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y * params.stride_y + global_id.x;
    if (i >= params.num_elements) {
        return;
    }

    image[i] = clamp(image[i], 0.0, 1.0);
}

@compute
@workgroup_size(256, 1, 1)
fn to_srgb(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y * params.stride_y + global_id.x;
    if (i >= params.num_elements) {
        return;
    }

    let x = clamp(image[i], 0.0, 1.0);
    var result: f32;
    if (x <= 0.0031308) {
        result = x * 12.92;
    } else {
        result = 1.055 * pow(x, 1.0 / 2.4) - 0.055;
    }
    image[i] = clamp(result, 0.0, 1.0);
}

@group(0) @binding(2)
var<storage, read> curve: array<f32>;
@group(0) @binding(3)
var<storage, read> mask: array<u32>;
@group(0) @binding(4)
var<storage, read> channels: array<u32>;

struct ToneCurveParams {
    channel_count: u32,
    num_pixels: u32,
    stride_y: u32,
};

@group(0) @binding(5)
var<uniform> tone_curve_params: ToneCurveParams;

@compute
@workgroup_size(256, 1, 1)
fn tone_curve_lut(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y * tone_curve_params.stride_y + global_id.x;
    if (i >= tone_curve_params.num_pixels) {
        return;
    }

    let base_idx = i * 3;
    let mask_idx = i;

    if (mask[mask_idx] != 0u) {
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

struct ToneCurveByHueParams {
    ch_hue: u32,
    ch_target: u32,
    num_pixels: u32,
    stride_y: u32,
};

@group(0) @binding(6)
var<uniform> tone_curve_by_hue_params: ToneCurveByHueParams;

@compute
@workgroup_size(256, 1, 1)
fn tone_curve_by_hue(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y * tone_curve_by_hue_params.stride_y + global_id.x;
    if (i >= tone_curve_by_hue_params.num_pixels) {
        return;
    }

    let idx = i * 3;

    if (mask[i] != 0u) {
        let hue = clamp(image[idx + tone_curve_by_hue_params.ch_hue], 0.0, 1.0);
        let val = clamp(image[idx + tone_curve_by_hue_params.ch_target], 0.0, 1.0);

        var hue_idx = u32(hue * 65535.0);
        hue_idx = clamp(hue_idx, 0u, 65535u);

        let gain = curve[hue_idx] / 65535.0;
        let new_val = clamp(val * gain, 0.0, 1.0);

        image[idx + tone_curve_by_hue_params.ch_target] = new_val;
    }
}

@compute
@workgroup_size(256, 1, 1)
fn rgb_to_hls(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y * params.stride_y + global_id.x;
    if (i >= params.num_elements / 3) {
        return;
    }

    let idx = i * 3;
    let r = clamp(image[idx + 0], 0.0, 1.0);
    let g = clamp(image[idx + 1], 0.0, 1.0);
    let b = clamp(image[idx + 2], 0.0, 1.0);

    let maxc = max(max(r, g), b);
    let minc = min(min(r, g), b);
    let delta = maxc - minc;

    let L = (maxc + minc) * 0.5;

    var S: f32 = 0.0;
    if (delta > 0.0) {
        if (L > 0.5) {
            S = delta / (2.0 - maxc - minc);
        } else {
            S = delta / (maxc + minc);
        }
    }

    var H: f32 = 0.0;
    if (delta > 0.0) {
        if (maxc == r) {
            H = (g - b) / delta;
            if (g < b) {
                H = H + 6.0;
            }
        } else if (maxc == g) {
            H = (b - r) / delta + 2.0;
        } else {
            H = (r - g) / delta + 4.0;
        }
        H = H / 6.0;
    }

    image[idx + 0] = clamp(H, 0.0, 1.0);
    image[idx + 1] = clamp(L, 0.0, 1.0);
    image[idx + 2] = clamp(S, 0.0, 1.0);
}

fn hue2rgb(p: f32, q: f32, t_in: f32) -> f32 {
    var t = t_in;
    t = t % 1.0;
    if (t < 0.0) {
        t = t + 1.0;
    }
    if (t < 1.0 / 6.0) {
        return p + (q - p) * 6.0 * t;
    } else if (t < 1.0 / 2.0) {
        return q;
    } else if (t < 2.0 / 3.0) {
        return p + (q - p) * (4.0 - 6.0 * t);
    } else {
        return p;
    }
}

@compute
@workgroup_size(256, 1, 1)
fn hls_to_rgb(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y * params.stride_y + global_id.x;
    if (i >= params.num_elements / 3) {
        return;
    }

    let idx = i * 3;
    let H = clamp(image[idx + 0], 0.0, 1.0);
    let L = clamp(image[idx + 1], 0.0, 1.0);
    let S = clamp(image[idx + 2], 0.0, 1.0);

    var R: f32;
    var G: f32;
    var B: f32;
    if (S == 0.0) {
        R = L;
        G = L;
        B = L;
    } else {
        var q: f32;
        if (L < 0.5) {
            q = L * (1.0 + S);
        } else {
            q = (L + S - L * S);
        }
        let p = 2.0 * L - q;
        R = hue2rgb(p, q, H + 1.0 / 3.0);
        G = hue2rgb(p, q, H);
        B = hue2rgb(p, q, H - 1.0 / 3.0);
    }

    image[idx + 0] = clamp(R, 0.0, 1.0);
    image[idx + 1] = clamp(G, 0.0, 1.0);
    image[idx + 2] = clamp(B, 0.0, 1.0);
}

struct WhiteBalanceParams {
    num_pixels: u32,
    r_gain: f32,
    g_gain: f32,
    b_gain: f32,
    stride_y: u32,
};

@group(0) @binding(7)
var<uniform> wb_params: WhiteBalanceParams;

@compute
@workgroup_size(256, 1, 1)
fn white_balance(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y * wb_params.stride_y + global_id.x;
    if (i >= wb_params.num_pixels) {
        return;
    }

    let idx = i * 3;

    if (mask[i] != 0u) {
        var r = clamp(image[idx + 0], 0.0, 1.0);
        var g = clamp(image[idx + 1], 0.0, 1.0);
        var b = clamp(image[idx + 2], 0.0, 1.0);

        r = r * wb_params.r_gain;
        g = g * wb_params.g_gain;
        b = b * wb_params.b_gain;

        image[idx + 0] = clamp(r, 0.0, 1.0);
        image[idx + 1] = clamp(g, 0.0, 1.0);
        image[idx + 2] = clamp(b, 0.0, 1.0);
    }
}

struct VignetteParams {
    width: u32,
    height: u32,
    strength: f32,
    num_pixels: u32,
    stride_y: u32,
};

@group(0) @binding(8)
var<uniform> vignette_params: VignetteParams;

@compute
@workgroup_size(256, 1, 1)
fn vignette_effect(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y * vignette_params.stride_y + global_id.x;
    if (i >= vignette_params.num_pixels) {
        return;
    }

    if (mask[i] != 0u) {
        let x = i % vignette_params.width;
        let y = i / vignette_params.width;

        let cx = f32(vignette_params.width) * 0.5;
        let cy = f32(vignette_params.height) * 0.5;
        let dx = (f32(x) - cx) / cx;
        let dy = (f32(y) - cy) / cy;

        let dist = sqrt(dx * dx + dy * dy);

        let inner_radius = 0.25;
        let adjusted_dist = max(0.0, (dist - inner_radius) / (1.0 - inner_radius));

        let t = clamp(adjusted_dist, 0.0, 1.0);
        let smooth_t = t * t * (3.0 - 2.0 * t);
        let falloff = pow(smooth_t, 2.0);

        var vignette = 1.0 - (vignette_params.strength * falloff);
        vignette = clamp(vignette, 0.0, 2.0);

        let idx = i * 3;
        image[idx + 0] = clamp(image[idx + 0] * vignette, 0.0, 1.0);
        image[idx + 1] = clamp(image[idx + 1] * vignette, 0.0, 1.0);
        image[idx + 2] = clamp(image[idx + 2] * vignette, 0.0, 1.0);
    }
}
