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
void tone_curve_by_oklch_h(
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
        // ch_hue is 2 for Oklch H
        let hue = clamp(image[idx + ch_hue], 0.0, 1.0); 
        let val = clamp(image[idx + ch_target], 0.0, 1.0);

        uint hue_idx = (uint)(hue * 65535.0);
        hue_idx = clamp(hue_idx, 0, 65535);

        let gain = curve[hue_idx] / 65535.0;
        let new_val = clamp(val * gain, 0.0, 1.0);

        image[idx + ch_target] = new_val;
    }
}

// --- OKLCH Conversion Intrinsics ---

// Matrix to convert from Linear sRGB to LMS cone space
static const float3x3 srgb_to_lms = float3x3(
    0.4121656120, 0.5362752080, 0.0514575653,
    0.2118591070, 0.6807189584, 0.1074065553,
    0.0883097947, 0.2818474174, 0.6302613616
);

// Matrix to convert from LMS to Oklab
static const float3x3 lms_to_oklab_m = float3x3(
    0.2104542553,  0.7936177850, -0.0040720468,
    1.9779984951, -2.4285922050,  0.4505937099,
    0.0259040371,  0.7827717662, -0.8086757660
);

// Matrix to convert from Oklab to LMS
static const float3x3 oklab_to_lms_m = float3x3(
    1.0,  0.3963377774,  0.2158037573,
    1.0, -0.1055613458, -0.0638541728,
    1.0, -0.0894841775, -1.2914855480
);

// Matrix to convert from LMS to Linear sRGB
static const float3x3 lms_to_srgb = float3x3(
    4.0767245293, -3.3072168827,  0.2307590544,
   -1.2681437731,  2.6093323231, -0.3411344290,
   -0.0041119885, -0.7034763098,  1.7068625689
);


float3 linear_srgb_to_lms(float3 c) {
    return mul(srgb_to_lms, c);
}

float3 lms_to_oklab(float3 c) {
    let l = pow(c.x, 1.0/3.0);
    let m = pow(c.y, 1.0/3.0);
    let s = pow(c.z, 1.0/3.0);
    return mul(lms_to_oklab_m, float3(l, m, s));
}

float3 oklab_to_lms(float3 c) {
    let l_ = c.x;
    let m_ = c.y;
    let s_ = c.z;
    let lms_ = mul(oklab_to_lms_m, float3(l_, m_, s_));
    return float3(
        lms_.x * lms_.x * lms_.x,
        lms_.y * lms_.y * lms_.y,
        lms_.z * lms_.z * lms_.z
    );
}

float3 lms_to_linear_srgb(float3 c) {
    return mul(lms_to_srgb, c);
}


[shader("compute")]
[numthreads(256, 1, 1)]
void linear_srgb_to_oklch(
    uint3 tid : SV_DispatchThreadID,
    uniform uint num_pixels,
    RWStructuredBuffer<float> image
) {
    let i = tid.x;
    if (i >= num_pixels) {
        return;
    }

    let idx = i * 3;
    let r = image[idx + 0];
    let g = image[idx + 1];
    let b = image[idx + 2];

    let lms = linear_srgb_to_lms(float3(r, g, b));

    // LMS to Lab
    let lab = lms_to_oklab(lms);
    let L = lab.x;
    let a = lab.y;
    let b_ = lab.z;

    // Lab to LCH
    let C = sqrt(a * a + b_ * b_);
    var H = atan2(b_, a);

    // Normalize H to 0-1 range
    H = H * (1.0 / (2.0 * 3.14159265359));
    if (H < 0.0) {
        H += 1.0;
    }

    image[idx + 0] = L;       // L: 0..1
    image[idx + 1] = C;       // C: 0..~0.4, will be normalized later if needed
    image[idx + 2] = H;       // H: 0..1
}


[shader("compute")]
[numthreads(256, 1, 1)]
void oklch_to_linear_srgb(
    uint3 tid : SV_DispatchThreadID,
    uniform uint num_pixels,
    RWStructuredBuffer<float> image
) {
    let i = tid.x;
    if (i >= num_pixels) {
        return;
    }
    
    let idx = i * 3;
    let L = image[idx + 0];
    let C = image[idx + 1];
    let H = image[idx + 2];

    // LCH to Lab
    let h_rad = H * (2.0 * 3.14159265359);
    let a = C * cos(h_rad);
    let b_ = C * sin(h_rad);

    let lab = float3(L, a, b_);

    // Lab to LMS
    let lms = oklab_to_lms(lab);
    
    // LMS to linear sRGB
    let rgb = lms_to_linear_srgb(lms);

    image[idx + 0] = rgb.x;
    image[idx + 1] = rgb.y;
    image[idx + 2] = rgb.z;
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