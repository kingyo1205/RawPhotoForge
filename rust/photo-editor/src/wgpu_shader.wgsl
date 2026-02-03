// wgpu_shader.wgsl

//--------------------------------------------------------------------------------
// Bindings
//--------------------------------------------------------------------------------
@group(0) @binding(0) var image_in: texture_2d<f32>;
@group(0) @binding(1) var image_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> base_params: BaseParams;
@group(0) @binding(3) var masks_tex: texture_2d_array<f32>;
@group(0) @binding(4) var<storage, read> masks_params: array<GpuEditParameters>;

@group(0) @binding(5) var<storage, read> tone_lut: array<i32>;
@group(0) @binding(6) var<storage, read> brightness_curve: array<i32>;
@group(0) @binding(7) var<storage, read> hue_curve: array<i32>;
@group(0) @binding(8) var<storage, read> saturation_curve: array<i32>;
@group(0) @binding(9) var<storage, read> lightness_curve: array<i32>;

struct BaseParams {
    width: u32,
    height: u32,
    num_masks: u32
};

struct GpuEditParameters {
    r_gain: f32,
    g_gain: f32,
    b_gain: f32,
    vignette: i32,
    lens_distortion: f32,
};




//--------------------------------------------------------------------------------
// Color Space Conversions (vec4)
//--------------------------------------------------------------------------------

const M1 = mat3x3<f32>(
    0.4122214708, 0.2119034982, 0.0883024619,
    0.5363325363, 0.6806995451, 0.2817188376,
    0.0514459929, 0.1073969566, 0.6299787005
);

const M1_INV = mat3x3<f32>(
    4.0767416621, -1.2684380046, -0.0041960863,
    -3.3077115913, 2.6097574011, -0.7034186147,
    0.2309699292, -0.3413193965, 1.7076147010
);

const M2 = mat3x3<f32>(
    0.2104542553, 1.9779984951, 0.0259040371,
    0.7936177850, -2.4285922050, 0.7827717662,
    -0.0040720468, 0.4505937099, -0.8086757660
);

const M2_INV = mat3x3<f32>(
    1.0, 1.0, 1.0,
    0.3963377774, -0.1055613458, -0.089484177,
    0.2158037573, -0.0638541728, -1.2914855480
);

fn linear_srgb_to_oklch(c: vec4<f32>) -> vec4<f32> {
    let lms = M1 * c.rgb;
    // pow(x, 1/3) with protection against negative/zero
    let lms_cbrt = pow(max(lms, vec3<f32>(0.0)), vec3<f32>(1.0 / 3.0));
    let oklab = M2 * lms_cbrt;
    
    let L = oklab.x;
    let C = sqrt(oklab.y * oklab.y + oklab.z * oklab.z);
    var h = atan2(oklab.z, oklab.y) / (2.0 * 3.14159265359);
    if (h < 0.0) { h += 1.0; }
    return vec4<f32>(L, C, h, 1.0);
}

fn oklch_to_linear_srgb(c: vec4<f32>) -> vec4<f32> {
    let h = c.z * 2.0 * 3.14159265359;
    let oklab = vec3<f32>(c.x, c.y * cos(h), c.y * sin(h));
    
    let lms_cbrt = M2_INV * oklab;
    let lms = lms_cbrt * lms_cbrt * lms_cbrt;
    return vec4<f32>(M1_INV * lms, 1.0);
}
fn srgb_to_linear(c: vec4<f32>) -> vec4<f32> {
    let rgb = c.rgb;
    let linear_rgb = select(
        pow((rgb + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)),
        rgb / 12.92,
        rgb <= vec3<f32>(0.04045)
    );
    return vec4<f32>(linear_rgb, 1.0);
}

fn linear_to_srgb(c: vec4<f32>) -> vec4<f32> {
    let rgb = c.rgb;
    let srgb = select(
        1.055 * pow(rgb, vec3<f32>(1.0 / 2.4)) - 0.055,
        rgb * 12.92,
        rgb <= vec3<f32>(0.0031308)
    );
    return vec4<f32>(srgb, 1.0);
}

//--------------------------------------------------------------------------------
// Effects
//--------------------------------------------------------------------------------

fn lens_distortion_sample(xy: vec2<i32>, distortion: f32) -> vec4<f32> {
    let strength = -0.5 * (distortion / 100.0);

    let w_u: u32 = base_params.width;
    let h_u: u32 = base_params.height;

    let w: f32 = f32(w_u);
    let h: f32 = f32(h_u);

    // そのまま(歪みなし)なら最速で返す
    if (strength == 0.0) {
        return textureLoad(image_in, xy, 0);
    }

    // UV(0..1)
    let uv = vec2<f32>(f32(xy.x) / w, f32(xy.y) / h);

    // 歪み補正
    let centered_uv = uv - 0.5;
    let aspect = w / h;

    var corrected_uv = centered_uv * vec2<f32>(aspect, 1.0);

    let r2 = dot(corrected_uv, corrected_uv);
    let distorted = corrected_uv / (1.0 + strength * r2);
    let final_uv = (distorted / vec2<f32>(aspect, 1.0)) + 0.5;

    // 範囲外は黒
    if (any(final_uv < vec2<f32>(0.0)) || any(final_uv > vec2<f32>(1.0))) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let px = final_uv.x * (w - 1.0);
    let py = final_uv.y * (h - 1.0);

    let x0_f = floor(px);
    let y0_f = floor(py);

    let x0_i: i32 = i32(x0_f);
    let y0_i: i32 = i32(y0_f);


    let x1_i: i32 = min(x0_i + 1, i32(w_u) - 1);
    let y1_i: i32 = min(y0_i + 1, i32(h_u) - 1);

    let tx: f32 = px - x0_f;
    let ty: f32 = py - y0_f;


    let c00 = textureLoad(image_in, vec2<i32>(x0_i, y0_i), 0);
    let c10 = textureLoad(image_in, vec2<i32>(x1_i, y0_i), 0);
    let c01 = textureLoad(image_in, vec2<i32>(x0_i, y1_i), 0);
    let c11 = textureLoad(image_in, vec2<i32>(x1_i, y1_i), 0);


    let cx0 = mix(c00, c10, tx);
    let cx1 = mix(c01, c11, tx);
    return mix(cx0, cx1, ty);
}

fn vignette(rgb_vec4: vec4<f32>, vignette_value: i32, xy: vec2<i32>) -> vec4<f32>{
    let vign_strength = (-f32(vignette_value) / 100.0) * 2.0;
    if (vign_strength != 0.0) {
        let coord = (vec2<f32>(xy) / vec2<f32>(f32(base_params.width), f32(base_params.height)) - 0.5) * 2.0;
        let dist = length(coord);
        let falloff = pow(clamp((dist - 0.25) / 0.75, 0.0, 1.0), 4.0);
        let rgb_vec4 = vec4<f32>(rgb_vec4.rgb * clamp(1.0 - (vign_strength * falloff), 0.0, 2.0), 1.0);

        return rgb_vec4;
    } else {
        return rgb_vec4;
    }
} 


//--------------------------------------------------------------------------------
// Curve
//--------------------------------------------------------------------------------

fn lut_fetch(which: u32, mask_index: u32, idx: u32) -> u32 {
    let base = mask_index * 65536u + idx;

    switch(which) {
        case 0u: { return u32(clamp(tone_lut[base], 0, 65535)); }
        case 1u: { return u32(clamp(brightness_curve[base], 0, 65535)); }
        case 2u: { return u32(clamp(hue_curve[base], 0, 65535)); }
        case 3u: { return u32(clamp(saturation_curve[base], 0, 65535)); }
        case 4u: { return u32(clamp(lightness_curve[base], 0, 65535)); }
        default  { return 0; }
    }
}

//--------------------------------------------------------------------------------
// Main
//--------------------------------------------------------------------------------

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let xy = vec2<i32>(gid.xy);
    if (gid.x >= base_params.width || gid.y >= base_params.height) { return; }

    let main_p = masks_params[0];

    // Lens Distortion (Main only)
    var col = lens_distortion_sample(xy, main_p.lens_distortion);
    // srgb to linear
    var rgb_vec4 = srgb_to_linear(col);

    // Vignette (Main only)
    rgb_vec4 = vignette(rgb_vec4, main_p.vignette, xy);

    // Per-mask Linear RGB adjustments
    for (var mask_index = 0u; mask_index < base_params.num_masks; mask_index++) {
        let mask_value = textureLoad(masks_tex, xy, mask_index, 0).r;
        if (mask_value != 1.0) { continue; }

        let p = masks_params[mask_index];
        
        // White Balance
        var r_f32 = rgb_vec4.r * p.r_gain;
        var g_f32 = rgb_vec4.g * p.g_gain;
        var b_f32 = rgb_vec4.b * p.b_gain;

        var r_u32 = u32(r_f32 * 65535);
        var g_u32 = u32(g_f32 * 65535);
        var b_u32 = u32(b_f32 * 65535);
        // Curves

        // tone
        r_u32 = lut_fetch(0u, mask_index, r_u32);
        g_u32 = lut_fetch(0u, mask_index, g_u32);
        b_u32 = lut_fetch(0u, mask_index, b_u32);

        // brightness
        r_u32 = lut_fetch(1u, mask_index, r_u32);
        g_u32 = lut_fetch(1u, mask_index, g_u32);
        b_u32 = lut_fetch(1u, mask_index, b_u32);

        r_f32 = f32(r_u32) / 65535.0;
        g_f32 = f32(g_u32) / 65535.0;
        b_f32 = f32(b_u32) / 65535.0;

        rgb_vec4 = vec4<f32>(r_f32, g_f32, b_f32, 1.0);
    }

    // Per-mask OKLCH adjustments
    var oklch_vec4 = linear_srgb_to_oklch(rgb_vec4);
    for (var mask_index = 0u; mask_index < base_params.num_masks; mask_index++) {
        let mask_value = textureLoad(masks_tex, xy, mask_index, 0).r;
        if (mask_value != 1.0) { continue; }

        var l_f32 = oklch_vec4.x;
        var c_f32 = oklch_vec4.y;
        var h_f32 = oklch_vec4.z;

        var l_u32 = u32(l_f32 * 65535);
        var c_u32 = u32(c_f32 * 65535);
        var h_u32 = u32(h_f32 * 65535);
        
        let new_hue = lut_fetch(2u, mask_index, h_u32);
        let saturation_gain = lut_fetch(3u, mask_index, h_u32);
        let lightness_gain = lut_fetch(4u, mask_index, h_u32);

        oklch_vec4.z = f32(new_hue) / 65535.0;
        oklch_vec4.y = c_f32 * (f32(saturation_gain) / 32767.5);
        oklch_vec4.x = l_f32 * (f32(lightness_gain) / 32767.5);
    }
    rgb_vec4 = oklch_to_linear_srgb(oklch_vec4);

    // Final Output
    let out_col = linear_to_srgb(rgb_vec4);
    textureStore(image_out, xy, clamp(out_col, vec4(0.0), vec4(1.0)));
}