// slang_kernel.slang

[shader("compute")]
[numthreads(256, 1, 1)]
void to_linear(
    uint3 tid : SV_DispatchThreadID,
    uniform uint num_elements,
    RWStructuredBuffer<float> image
) {
    let i = tid.x;
    if (i >= num_elements) {
        return;
    }

    let x = clamp(image[i], 0.0, 1.0);
    var result = (x <= 0.04045) ? (x / 12.92) : pow((x + 0.055) / 1.055, 2.4);
    image[i] = clamp(result, 0.0, 1.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clip_0_1(
    uint3 tid : SV_DispatchThreadID,
    uniform uint num_elements,
    RWStructuredBuffer<float> image
) {
    let i = tid.x;
    if (i >= num_elements) {
        return;
    }

    image[i] = clamp(image[i], 0.0, 1.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void to_srgb(
    uint3 tid : SV_DispatchThreadID,
    uniform uint num_elements,
    RWStructuredBuffer<float> image
) {
    let i = tid.x;
    if (i >= num_elements) {
        return;
    }

    let x = clamp(image[i], 0.0, 1.0);
    var result = (x <= 0.0031308) ? (x * 12.92) : (1.055 * pow(x, 1.0 / 2.4) - 0.055);
    image[i] = clamp(result, 0.0, 1.0);
}


[shader("compute")]
[numthreads(256, 1, 1)]
void tone_curve_lut(
    uint3 tid : SV_DispatchThreadID,
    uniform uint channel_count,
    uniform uint num_pixels,
    RWStructuredBuffer<float> image,
    StructuredBuffer<float> curve,
    StructuredBuffer<int8_t> mask,
    StructuredBuffer<uint> channels
) {
    let i = tid.x;
    if (i >= num_pixels) {
        return;
    }

    let base_idx = i * 3;
    let mask_idx = i;

    if (mask[mask_idx] != 0) {
        for (uint ch = 0; ch < channel_count; ch = ch + 1) {
            let c = channels[ch];
            let val = clamp(image[base_idx + c], 0.0, 1.0);
            uint lut_idx = (uint)(val * 65535.0);
            lut_idx = clamp(lut_idx, 0, 65535);
            let result = curve[lut_idx] / 65535.0;
            image[base_idx + c] = clamp(result, 0.0, 1.0);
        }
    }
}


[shader("compute")]
[numthreads(256, 1, 1)]
void tone_curve_by_hue(
    uint3 tid : SV_DispatchThreadID,
    uniform uint ch_hue,
    uniform uint ch_target,
    uniform uint num_pixels,
    RWStructuredBuffer<float> image,
    StructuredBuffer<float> curve,
    StructuredBuffer<int8_t> mask
) {
    let i = tid.x;
    if (i >= num_pixels) {
        return;
    }

    let idx = i * 3;

    if (mask[i] != 0) {
        let hue = clamp(image[idx + ch_hue], 0.0, 1.0);
        let val = clamp(image[idx + ch_target], 0.0, 1.0);

        uint hue_idx = (uint)(hue * 65535.0);
        hue_idx = clamp(hue_idx, 0, 65535);

        let gain = curve[hue_idx] / 65535.0;
        let new_val = clamp(val * gain, 0.0, 1.0);

        image[idx + ch_target] = new_val;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rgb_to_hls(
    uint3 tid : SV_DispatchThreadID,
    uniform uint num_pixels,
    RWStructuredBuffer<float> image
) {
    let i = tid.x;
    if (i >= num_pixels) {
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

    var S = 0.0;
    if (delta > 0.0) {
        if (L > 0.5) {
            S = delta / (2.0 - maxc - minc);
        } else {
            S = delta / (maxc + minc);
        }
    }

    var H = 0.0;
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

float hue2rgb(float p, float q, float t_in) {
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

[shader("compute")]
[numthreads(256, 1, 1)]
void hls_to_rgb(
    uint3 tid : SV_DispatchThreadID,
    uniform uint num_pixels,
    RWStructuredBuffer<float> image
) {
    let i = tid.x;
    if (i >= num_pixels) {
        return;
    }

    let idx = i * 3;
    let H = clamp(image[idx + 0], 0.0, 1.0);
    let L = clamp(image[idx + 1], 0.0, 1.0);
    let S = clamp(image[idx + 2], 0.0, 1.0);

    float R, G, B;
    if (S == 0.0) {
        R = L;
        G = L;
        B = L;
    } else {
        var q = (L < 0.5) ? (L * (1.0 + S)) : (L + S - L * S);
        let p = 2.0 * L - q;
        R = hue2rgb(p, q, H + 1.0 / 3.0);
        G = hue2rgb(p, q, H);
        B = hue2rgb(p, q, H - 1.0 / 3.0);
    }

    image[idx + 0] = clamp(R, 0.0, 1.0);
    image[idx + 1] = clamp(G, 0.0, 1.0);
    image[idx + 2] = clamp(B, 0.0, 1.0);
}


[shader("compute")]
[numthreads(256, 1, 1)]
void white_balance(
    uint3 tid : SV_DispatchThreadID,
    uniform uint num_pixels,
    uniform float r_gain,
    uniform float g_gain,
    uniform float b_gain,
    RWStructuredBuffer<float> image,
    StructuredBuffer<int8_t> mask
) {
    let i = tid.x;
    if (i >= num_pixels) {
        return;
    }

    let idx = i * 3;

    if (mask[i] != 0) {
        var r = clamp(image[idx + 0], 0.0, 1.0);
        var g = clamp(image[idx + 1], 0.0, 1.0);
        var b = clamp(image[idx + 2], 0.0, 1.0);

        r = r * r_gain;
        g = g * g_gain;
        b = b * b_gain;

        image[idx + 0] = clamp(r, 0.0, 1.0);
        image[idx + 1] = clamp(g, 0.0, 1.0);
        image[idx + 2] = clamp(b, 0.0, 1.0);
    }
}


[shader("compute")]
[numthreads(256, 1, 1)]
void vignette_effect(
    uint3 tid : SV_DispatchThreadID,
    uniform uint width,
    uniform uint height,
    uniform float strength,
    uniform uint num_pixels,
    RWStructuredBuffer<float> image,
    StructuredBuffer<int8_t> mask
) {
    let i = tid.x;
    if (i >= num_pixels) {
        return;
    }

    if (mask[i] != 0) {
        let x = i % width;
        let y = i / width;

        let cx = float(width) * 0.5;
        let cy = float(height) * 0.5;
        let dx = (float(x) - cx) / cx;
        let dy = (float(y) - cy) / cy;

        let dist = sqrt(dx * dx + dy * dy);

        let inner_radius = 0.25;
        let adjusted_dist = max(0.0, (dist - inner_radius) / (1.0 - inner_radius));

        let t = clamp(adjusted_dist, 0.0, 1.0);
        let smooth_t = t * t * (3.0 - 2.0 * t);
        let falloff = pow(smooth_t, 2.0);

        var vignette = 1.0 - (strength * falloff);
        vignette = clamp(vignette, 0.0, 2.0);

        let idx = i * 3;
        image[idx + 0] = clamp(image[idx + 0] * vignette, 0.0, 1.0);
        image[idx + 1] = clamp(image[idx + 1] * vignette, 0.0, 1.0);
        image[idx + 2] = clamp(image[idx + 2] * vignette, 0.0, 1.0);
    }
}