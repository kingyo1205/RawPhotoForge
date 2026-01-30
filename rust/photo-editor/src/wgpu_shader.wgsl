
//--------------------------------------------------------------------------------
// Bindings (Common for most kernels)
//--------------------------------------------------------------------------------
@group(0) @binding(0) var image_in: texture_2d<f32>;
@group(0) @binding(1) var image_out: texture_storage_2d<rgba32float, write>;

//--------------------------------------------------------------------------------
// Uniforms & Other Resources
//--------------------------------------------------------------------------------
struct BaseParams {
    width: u32,
    height: u32,
};
@group(0) @binding(2) var<uniform> base_params: BaseParams;

@group(0) @binding(3) var<storage, read> curve: array<f32>;

@group(0) @binding(4) var mask_texture: texture_2d<f32>;

@group(0) @binding(5) var<storage, read> channels: array<u32>;



struct ToneCurveParams {

    width: u32,

    height: u32,

    channel_count: u32,

};

@group(0) @binding(6) var<uniform> tone_curve_params: ToneCurveParams;



struct ToneCurveByHueParams {

    width: u32,

    height: u32,

    ch_hue: u32,

    ch_target: u32,

};

@group(0) @binding(7) var<uniform> tone_curve_by_hue_params: ToneCurveByHueParams;



struct WhiteBalanceParams {

    width: u32,

    height: u32,

    r_gain: f32,

    g_gain: f32,

    b_gain: f32,

};

@group(0) @binding(8) var<uniform> wb_params: WhiteBalanceParams;



struct VignetteParams {

    width: u32,

    height: u32,

    strength: f32,

};

@group(0) @binding(9) var<uniform> vignette_params: VignetteParams;



struct LensDistortionParams {

    width: u32,

    height: u32,

    strength: f32,

};

@group(0) @binding(10) var<uniform> lens_distortion_params: LensDistortionParams;



fn texture_lerp_from_texture(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {

    let dims = vec2<f32>(textureDimensions(tex));

    let tc = uv * (dims - 1.0);

    let p0 = floor(tc);

    let p1 = min(p0 + 1.0, dims - 1.0);

    let fract_p = fract(tc);



    let c00 = textureLoad(tex, vec2<u32>(u32(p0.x), u32(p0.y)), 0);

    let c01 = textureLoad(tex, vec2<u32>(u32(p1.x), u32(p0.y)), 0);

    let c10 = textureLoad(tex, vec2<u32>(u32(p0.x), u32(p1.y)), 0);

    let c11 = textureLoad(tex, vec2<u32>(u32(p1.x), u32(p1.y)), 0);

    

    let top = mix(c00, c01, fract_p.x);

    let bottom = mix(c10, c11, fract_p.x);

    return mix(top, bottom, fract_p.y);

}



//--------------------------------------------------------------------------------

// OKLCH Conversion Matrices and Functions
// OKLab/OKLCH conversion constants
// Source: Bjorn Ottosson (original OKLab definition)
// https://bottosson.github.io/posts/oklab/
//
// IMPORTANT:
// - Do NOT modify these coefficients.
// - Do NOT refactor/reformat/round constants.
// - Any change will alter color output and break matching with reference implementations.
//--------------------------------------------------------------------------------



const M1: mat3x3<f32> = mat3x3<f32>(

    0.4122214708, 0.2119034982, 0.0883024619,

    0.5363325363, 0.6806995451, 0.2817188376,

    0.0514459929, 0.1073969566, 0.6299787005

);



const M1_INV: mat3x3<f32> = mat3x3<f32>(

    4.0767416621, -1.2684380046, -0.0041960863,

    -3.3077115913, 2.6097574011, -0.7034186147,

    0.2309699292, -0.3413193965, 1.7076147010

);



const M2: mat3x3<f32> = mat3x3<f32>(

    0.2104542553, 1.9779984951, 0.0259040371,

    0.7936177850, -2.4285922050, 0.7827717662,

    -0.0040720468, 0.4505937099, -0.8086757660

);



const M2_INV: mat3x3<f32> = mat3x3<f32>(

    1.0,  1.0,  1.0,

    0.3963377774, -0.1055613458, -0.089484177,

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



fn check_bounds(global_id: vec2<u32>) -> bool {

    let dims = textureDimensions(image_out);

    return global_id.x >= dims.x || global_id.y >= dims.y;

}



@compute @workgroup_size(16, 16, 1)

fn to_linear(@builtin(global_invocation_id) global_id: vec3<u32>) {

    if (check_bounds(global_id.xy)) { return; }



    let color_in = textureLoad(image_in, global_id.xy, 0);

    var linear_rgb: vec3<f32>;

    for (var i = 0; i < 3; i = i + 1) {

        let x = color_in[i];

        if (x <= 0.04045) {

            linear_rgb[i] = x / 12.92;

        } else {

            linear_rgb[i] = pow((x + 0.055) / 1.055, 2.4);

        }

    }

    textureStore(image_out, global_id.xy, vec4(linear_rgb, color_in.a));

}



@compute @workgroup_size(16, 16, 1)

fn to_srgb(@builtin(global_invocation_id) global_id: vec3<u32>) {

    if (check_bounds(global_id.xy)) { return; }



    let color_in = textureLoad(image_in, global_id.xy, 0);

    var srgb: vec3<f32>;

    for (var i = 0; i < 3; i = i + 1) {

        let x = color_in[i];

        if (x <= 0.0031308) {

            srgb[i] = x * 12.92;

        } else {

            srgb[i] = 1.055 * pow(x, 1.0 / 2.4) - 0.055;

        }

    }

    textureStore(image_out, global_id.xy, vec4(srgb, color_in.a));

}



@compute @workgroup_size(16, 16, 1)

fn clip_0_1(@builtin(global_invocation_id) global_id: vec3<u32>) {

    if (check_bounds(global_id.xy)) { return; }

    

    let color_in = textureLoad(image_in, global_id.xy, 0);

    textureStore(image_out, global_id.xy, clamp(color_in, vec4(0.0), vec4(1.0)));

}



@compute @workgroup_size(16, 16, 1)

fn rgb_to_oklch(@builtin(global_invocation_id) global_id: vec3<u32>) {

    if (check_bounds(global_id.xy)) { return; }

    

    let rgb_linear = textureLoad(image_in, global_id.xy, 0).rgb;

    let oklab = linear_srgb_to_oklab(rgb_linear);

    let oklch = oklab_to_oklch(oklab);

    

    textureStore(image_out, global_id.xy, vec4(oklch, 1.0));

}



@compute @workgroup_size(16, 16, 1)

fn oklch_to_rgb(@builtin(global_invocation_id) global_id: vec3<u32>) {

    if (check_bounds(global_id.xy)) { return; }



    let oklch = textureLoad(image_in, global_id.xy, 0).rgb;

    let oklab = oklch_to_oklab(oklch);

    let rgb_linear = oklab_to_linear_srgb(oklab);



    textureStore(image_out, global_id.xy, vec4(rgb_linear, 1.0));

}



@compute @workgroup_size(16, 16, 1)

fn tone_curve_lut(@builtin(global_invocation_id) global_id: vec3<u32>) {

    if (check_bounds(global_id.xy)) { return; }



    let mask_val = textureLoad(mask_texture, global_id.xy, 0).r;

    if (mask_val == 0.0) {

        textureStore(image_out, global_id.xy, textureLoad(image_in, global_id.xy, 0));

        return;

    }



    var color = textureLoad(image_in, global_id.xy, 0);

    for (var i: u32 = 0; i < tone_curve_params.channel_count; i = i + 1) {

        let ch_idx = channels[i];

        let val = clamp(color[ch_idx], 0.0, 1.0);

        var lut_idx = u32(val * 65535.0);

        lut_idx = clamp(lut_idx, 0u, 65535u);

        let result = curve[lut_idx] / 65535.0;

        color[ch_idx] = mix(color[ch_idx], clamp(result, 0.0, 1.0), mask_val);

    }

    textureStore(image_out, global_id.xy, color);

}



@compute @workgroup_size(16, 16, 1)

fn tone_curve_by_hue(@builtin(global_invocation_id) global_id: vec3<u32>) {

    if (check_bounds(global_id.xy)) { return; }



    let mask_val = textureLoad(mask_texture, global_id.xy, 0).r;

    var color = textureLoad(image_in, global_id.xy, 0);



    if (mask_val > 0.0) {

        let hue = clamp(color[tone_curve_by_hue_params.ch_hue], 0.0, 1.0);

        let val = color[tone_curve_by_hue_params.ch_target];



        var hue_idx = u32(hue * 65535.0);

        hue_idx = clamp(hue_idx, 0u, 65535u);



        let gain = curve[hue_idx] / 65535.0;

        let new_val = val * gain;

        

        color[tone_curve_by_hue_params.ch_target] = mix(val, new_val, mask_val);

    }

    textureStore(image_out, global_id.xy, color);

}





@compute @workgroup_size(16, 16, 1)

fn white_balance(@builtin(global_invocation_id) global_id: vec3<u32>) {

    if (check_bounds(global_id.xy)) { return; }

    

    let mask_val = textureLoad(mask_texture, global_id.xy, 0).r;

    var color = textureLoad(image_in, global_id.xy, 0);



    if (mask_val > 0.0) {

        var adjusted_color = color.rgb;

        adjusted_color.r = adjusted_color.r * wb_params.r_gain;

        adjusted_color.g = adjusted_color.g * wb_params.g_gain;

        adjusted_color.b = adjusted_color.b * wb_params.b_gain;

        let mixed_rgb = mix(color.rgb, adjusted_color, mask_val);

        color.r = mixed_rgb.r;

        color.g = mixed_rgb.g;

        color.b = mixed_rgb.b;

    }

    textureStore(image_out, global_id.xy, color);

}



@compute @workgroup_size(16, 16, 1)

fn vignette_effect(@builtin(global_invocation_id) global_id: vec3<u32>) {

    if (check_bounds(global_id.xy)) { return; }



    let p = vignette_params;

    let mask_val = textureLoad(mask_texture, global_id.xy, 0).r;

    var color = textureLoad(image_in, global_id.xy, 0);



    if (mask_val > 0.0) {

        let cx = f32(p.width) * 0.5;

        let cy = f32(p.height) * 0.5;

        let dx = (f32(global_id.x) - cx) / cx;

        let dy = (f32(global_id.y) - cy) / cy;

        let dist = sqrt(dx * dx + dy * dy);



        let inner_radius = 0.25;

        let adjusted_dist = max(0.0, (dist - inner_radius) / (1.0 - inner_radius));



        let t = clamp(adjusted_dist, 0.0, 1.0);

        let smooth_t = t * t * (3.0 - 2.0 * t);

        let falloff = pow(smooth_t, 4.0);



        var vignette = 1.0 - (p.strength * falloff);

        vignette = clamp(vignette, 0.0, 2.0);



        let final_vignette = mix(1.0, vignette, mask_val);

                let vignette_rgb = color.rgb * final_vignette;

        color.r = vignette_rgb.r;

        color.g = vignette_rgb.g;

        color.b = vignette_rgb.b;

    }

    textureStore(image_out, global_id.xy, color);

}





@compute @workgroup_size(16, 16, 1)

fn lens_distortion_effect(@builtin(global_invocation_id) global_id: vec3<u32>) {

    let p = lens_distortion_params;

    if (check_bounds(global_id.xy)) { return; }



    let dims = textureDimensions(image_in);

    let uv = vec2<f32>(f32(global_id.x) / f32(dims.x - 1u), f32(global_id.y) / f32(dims.y - 1u));

    

    let centered_uv = uv - 0.5;



    let aspect_ratio = f32(p.width) / f32(p.height);

    var corrected_uv = centered_uv;

    corrected_uv.x = corrected_uv.x * aspect_ratio;



    let r2 = dot(corrected_uv, corrected_uv);

    let k = p.strength;

    let distorted_uv_centered = corrected_uv / (1.0 + k * r2);

    

    var final_uv_centered = distorted_uv_centered;

    final_uv_centered.x = final_uv_centered.x / aspect_ratio;

    

    let final_uv = final_uv_centered + 0.5;

    

    var final_color = vec4(0.0, 0.0, 0.0, 1.0);

    if (final_uv.x >= 0.0 && final_uv.x <= 1.0 && final_uv.y >= 0.0 && final_uv.y <= 1.0) {

        final_color = texture_lerp_from_texture(image_in, final_uv);

    }

    

    textureStore(image_out, global_id.xy, final_color);

}