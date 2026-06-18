
export function pchipInterpolate(
    xPts: number[] | Float32Array,
    yPts: number[] | Float32Array,
    xEval: number[] | Float32Array,
): Float32Array {
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
        if (h_i <= 0.0) {
            throw new Error(`x_pts must be strictly increasing, but found a non-increasing value at index ${i}`);
        }
        h[i] = h_i;
        del[i] = (yPts[i + 1] - yPts[i]) / h_i;
    }


    const slopes = new Float32Array(n);


    slopes[0] = del[0];
    slopes[n - 1] = del[n - 2];


    for (let i = 1; i < n - 1; i++) {
        if (del[i - 1] * del[i] <= 0.0) {
            slopes[i] = 0.0;
        } else {
            const w1 = 2.0 * h[i] + h[i - 1];
            const w2 = h[i] + 2.0 * h[i - 1];
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



        let i = Array.prototype.findIndex.call(xPts, (p: number) => p > x);
        if (i === -1) {


            i = n - 1;
        }
        i = Math.max(0, i - 1);

        i = Math.min(i, n - 2);


        const h_val = h[i];
        const t = (x - xPts[i]) / h_val;
        const t2 = t * t;
        const t3 = t2 * t;


        const h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        const h10 = t3 - 2.0 * t2 + t;
        const h01 = -2.0 * t3 + 3.0 * t2;
        const h11 = t3 - t2;

        const y0 = yPts[i];
        const y1 = yPts[i + 1];
        const m0 = slopes[i];
        const m1 = slopes[i + 1];

        yEval[k] = h00 * y0 + h10 * h_val * m0 + h01 * y1 + h11 * h_val * m1;
    }

    return yEval;
}
