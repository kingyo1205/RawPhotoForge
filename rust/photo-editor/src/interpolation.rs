// interpolation.rs

use ndarray::Array1;
use num_traits::{AsPrimitive, FromPrimitive};
use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use crate::errors::InterpolationError;

pub fn pchip_interpolate<T>(
    x_pts: &Array1<T>,
    y_pts: &Array1<T>,
    x_eval: &Array1<T>,
) -> Result<Array1<T>, InterpolationError>
where
    T: Copy + PartialOrd + AsPrimitive<f32> + FromPrimitive + Debug,
    f32: AsPrimitive<T>,
    for<'a> &'a T: Sub<&'a T, Output = T>,
    for<'a> T: Div<T, Output = T> + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T>,
{
    if x_pts.len() != y_pts.len() {
        return Err(InterpolationError::MismatchedLengths {
            x_len: x_pts.len(),
            y_len: y_pts.len(),
        });
    }
    if x_pts.len() < 2 {
        return Err(InterpolationError::NotEnoughPoints {
            points: x_pts.len(),
        });
    }

    // 内部計算はf32で行う (GodotのPackedFloat32Arrayと挙動を合わせるため)
    let x_pts_f32: Array1<f32> = x_pts.mapv(|v| v.as_());
    let y_pts_f32: Array1<f32> = y_pts.mapv(|v| v.as_());
    let x_eval_f32: Array1<f32> = x_eval.mapv(|v| v.as_());

    let n = x_pts_f32.len();
    let mut y_eval = Array1::<f32>::zeros(x_eval_f32.len());

    // 1. 各区間の傾き (secants) と区間幅 (h) を計算
    let mut del = Vec::with_capacity(n - 1);
    let mut h = Vec::with_capacity(n - 1);

    for i in 0..(n - 1) {
        let h_i = x_pts_f32[i + 1] - x_pts_f32[i];
        if h_i <= 0.0 {
            // 単調増加でない場合はエラー
            return Err(InterpolationError::NotStrictlyIncreasing { index: i });
        }
        h.push(h_i);
        del.push((y_pts_f32[i + 1] - y_pts_f32[i]) / h_i);
    }

    // 2. 点ごとの傾き (slopes) を計算 (Harmonic Mean)
    let mut slopes = Array1::<f32>::zeros(n);

    // 両端点 (One-sided difference)
    slopes[0] = del[0];
    slopes[n - 1] = del[n - 2];

    // 中間点 (PCHIP logic)
    for i in 1..(n - 1) {
        // Godot: if s1 * s2 <= 0.0
        if del[i - 1] * del[i] <= 0.0 {
            slopes[i] = 0.0;
        } else {
            // Godot: w1 = 2 * dx2 + dx1
            let w1 = 2.0 * h[i] + h[i - 1];
            // Godot: w2 = dx2 + 2 * dx1
            let w2 = h[i] + 2.0 * h[i - 1];

            // Godot: (w1 + w2) / (w1 / s1 + w2 / s2)
            slopes[i] = (w1 + w2) / (w1 / del[i - 1] + w2 / del[i]);
        }
    }

    // 3. 評価点の補完計算
    for (k, &x) in x_eval_f32.iter().enumerate() {
        // 範囲外の処理 (Godotに合わせてClamp的な挙動)
        if x <= x_pts_f32[0] {
            y_eval[k] = y_pts_f32[0];
            continue;
        }
        if x >= x_pts_f32[n - 1] {
            y_eval[k] = y_pts_f32[n - 1];
            continue;
        }

        // 区間探索
        let i = match x_pts_f32
            .as_slice()
            .unwrap()
            .binary_search_by(|p| p.partial_cmp(&x).unwrap_or(Ordering::Equal))
        {
            Ok(idx) => idx,
            Err(idx) => idx - 1,
        };
        // 安全策
        let i = i.min(n - 2);

        let h_val = h[i];
        let t = (x - x_pts_f32[i]) / h_val;
        let t2 = t * t;
        let t3 = t2 * t;

        // エルミート基底関数
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        let y0 = y_pts_f32[i];
        let y1 = y_pts_f32[i + 1];
        let m0 = slopes[i];
        let m1 = slopes[i + 1];

        y_eval[k] = h00 * y0 + h10 * h_val * m0 + h01 * y1 + h11 * h_val * m1;
    }

    Ok(y_eval.mapv(|v| v.as_()))
}
