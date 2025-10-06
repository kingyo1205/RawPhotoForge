// raw_image_editor.opencl_kernel.cl


__kernel void to_linear(__global float *image, int num_elements) {
    int i = get_global_id(0);
    if (i >= num_elements) return;

    float x = clamp(image[i], 0.0f, 1.0f);
    if (x <= 0.04045f)
        image[i] = x / 12.92f;
    else
        image[i] = pow((x + 0.055f) / 1.055f, 2.4f);
}

__kernel void clip_0_1(__global float *arr, int num_elements) {
    int i = get_global_id(0);
    if (i >= num_elements) return;

    arr[i] = clamp(arr[i], 0.0f, 1.0f);
}

__kernel void to_srgb(__global float *image, int num_elements) {
    int i = get_global_id(0);
    if (i >= num_elements) return;

    float x = clamp(image[i], 0.0f, 1.0f);
    if (x <= 0.0031308f)
        image[i] = x * 12.92f;
    else
        image[i] = 1.055f * pow(x, 1.0f / 2.4f) - 0.055f;
}

// image: (num_pixels*3), flatten RGB [0,1]
// mask: (num_pixels), uchar 0 or 1
// curve: (65536), LUT
// channels: (channel_count), e.g. [0,1,2]
// channel_count: int
__kernel void tone_curve_lut(
    __global float *image,
    __global float *curve,
    __global uchar *mask,
    __global int *channels,
    int channel_count,
    int num_pixels
) {
    int i = get_global_id(0);
    if (i >= num_pixels) return;

    int base_idx = i * 3;
    int mask_idx = i;

    if (mask[mask_idx]) {
        for (int ch = 0; ch < channel_count; ch++) {
            int c = channels[ch];
            float val = clamp(image[base_idx + c], 0.0f, 1.0f);
            int lut_idx = (int)(val * 65535.0f);
            lut_idx = clamp(lut_idx, 0, 65535);
            image[base_idx + c] = curve[lut_idx] / 65535.0f;
        }
    }
}

// hls_image: (num_pixels*3), flatten HLS [0,1]
// mask: (num_pixels), uchar 0 or 1
// curve: (65536), LUT
__kernel void tone_curve_by_hue(
    __global float *hls_image,
    __global float *curve,
    __global uchar *mask,
    int ch_hue,
    int ch_target,
    int num_pixels
) {
    int i = get_global_id(0);
    if (i >= num_pixels) return;

    int idx = i * 3;

    if (mask[i]) {
        float hue = clamp(hls_image[idx + ch_hue], 0.0f, 1.0f);
        float val = clamp(hls_image[idx + ch_target], 0.0f, 1.0f);

        int hue_idx = (int)(hue * 65535.0f);
        hue_idx = clamp(hue_idx, 0, 65535);

        float gain = curve[hue_idx] / 65535.0f;
        float new_val = clamp(val * gain, 0.0f, 1.0f);

        hls_image[idx + ch_target] = new_val;
    }
}

// img: (num_pixels*3), flatten RGB [0,1]
__kernel void rgb_to_hls(
    __global float *img,
    int num_pixels
) {
    int i = get_global_id(0);
    if (i >= num_pixels) return;

    int idx = i * 3;
    float r = img[idx + 0];
    float g = img[idx + 1];
    float b = img[idx + 2];

    float maxc = fmax(fmax(r, g), b);
    float minc = fmin(fmin(r, g), b);
    float delta = maxc - minc;

    float L = (maxc + minc) * 0.5f;

    float S = 0.0f;
    if (delta > 0.0f) {
        if (L > 0.5f)
            S = delta / (2.0f - maxc - minc);
        else
            S = delta / (maxc + minc);
    }

    float H = 0.0f;
    if (delta > 0.0f) {
        if (maxc == r) {
            H = (g - b) / delta + (g < b ? 6.0f : 0.0f);
        } else if (maxc == g) {
            H = (b - r) / delta + 2.0f;
        } else {
            H = (r - g) / delta + 4.0f;
        }
        H /= 6.0f;
    }

    img[idx + 0] = H;
    img[idx + 1] = L;
    img[idx + 2] = S;
}

inline float hue2rgb(float p, float q, float t) {
    t = fmod(t, 1.0f);
    if (t < 0.0f) t += 1.0f;
    if (t < 1.0f/6.0f)      return p + (q - p) * 6.0f * t;
    else if (t < 1.0f/2.0f) return q;
    else if (t < 2.0f/3.0f) return p + (q - p) * (4.0f - 6.0f * t);
    else                    return p;
}

// img: (num_pixels*3), flatten HLS [0,1]
__kernel void hls_to_rgb(
    __global float *img,
    int num_pixels
) {
    int i = get_global_id(0);
    if (i >= num_pixels) return;

    int idx = i * 3;
    float H = img[idx + 0];
    float L = img[idx + 1];
    float S = img[idx + 2];

    float R, G, B;
    if (S == 0.0f) {
        R = G = B = L;
    } else {
        float q = (L < 0.5f) ? L * (1.0f + S) : (L + S - L * S);
        float p = 2.0f * L - q;
        R = hue2rgb(p, q, H + 1.0f/3.0f);
        G = hue2rgb(p, q, H);
        B = hue2rgb(p, q, H - 1.0f/3.0f);
    }

    img[idx + 0] = clamp(R, 0.0f, 1.0f);
    img[idx + 1] = clamp(G, 0.0f, 1.0f);
    img[idx + 2] = clamp(B, 0.0f, 1.0f);
}

// image: (num_pixels*3), flatten RGB [0,1]
// mask: (num_pixels), uchar 0 or 1
__kernel void white_balance(
    __global float *image,
    __global uchar *mask,
    int num_pixels,
    float r_gain,
    float g_gain,
    float b_gain
) {
    int i = get_global_id(0);
    if (i >= num_pixels) return;

    int idx = i * 3;

    if (mask[i]) {
        image[idx + 0] *= r_gain;
        image[idx + 1] *= g_gain;
        image[idx + 2] *= b_gain;
    }
}
