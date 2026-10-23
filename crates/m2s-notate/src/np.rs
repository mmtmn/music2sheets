//! Numeric helpers that reproduce numpy / Python semantics exactly where decisions depend on
//! them (pairwise summation, percentile interpolation, interp, round-half-even).

/// numpy's pairwise summation (what `np.add.reduce` does along a contiguous 1-D axis).
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// `np.mean` of a 1-D float array.
pub fn mean(a: &[f64]) -> f64 {
    pairwise_sum(a) / a.len() as f64
}

/// `np.median`.
pub fn median(a: &[f64]) -> f64 {
    let mut v = a.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// `np.percentile(a, q)` with the default linear method, including numpy's `_lerp` detail.
pub fn percentile(a: &[f64], q: f64) -> f64 {
    let mut v = a.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = v.len();
    let qq = q / 100.0;
    // numpy: n * q + (alpha + q * (1 - alpha - beta)) - 1 with alpha = beta = 1
    let virt = (n as f64 * qq + (1.0 - qq)) - 1.0;
    let (prev, next) = if virt >= (n - 1) as f64 {
        (n - 1, n - 1)
    } else if virt < 0.0 {
        (0, 0)
    } else {
        let p = virt.floor() as usize;
        (p, p + 1)
    };
    let gamma = if virt >= (n - 1) as f64 || virt < 0.0 {
        0.0
    } else {
        virt - prev as f64
    };
    let (lo, hi) = (v[prev], v[next.min(n - 1)]);
    let diff = hi - lo;
    if gamma >= 0.5 {
        hi - diff * (1.0 - gamma)
    } else {
        lo + diff * gamma
    }
}

/// `np.percentile` on integers (pitches).
pub fn percentile_i(a: &[i32], q: f64) -> f64 {
    let v: Vec<f64> = a.iter().map(|&x| x as f64).collect();
    percentile(&v, q)
}

/// `np.diff`.
pub fn diff(a: &[f64]) -> Vec<f64> {
    a.windows(2).map(|w| w[1] - w[0]).collect()
}

/// `np.interp(x, xp, fp)` for one x (xp increasing).
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    if x < xp[0] {
        return fp[0];
    }
    // j: last index with xp[j] <= x
    let j = xp.partition_point(|&v| v <= x) - 1;
    if j == n - 1 || xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let r = slope * (x - xp[j]) + fp[j];
    if r.is_nan() {
        let r2 = slope * (x - xp[j + 1]) + fp[j + 1];
        if r2.is_nan() && fp[j] == fp[j + 1] {
            return fp[j];
        }
        return r2;
    }
    r
}

/// `np.searchsorted(a, v, side)`.
pub fn searchsorted(a: &[f64], v: f64, right: bool) -> usize {
    if right {
        a.partition_point(|&x| x <= v)
    } else {
        a.partition_point(|&x| x < v)
    }
}

/// Python `round(x)` (half to even) as an integer.
pub fn round_i(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// Python `round(x, n)`.
pub fn round_n(x: f64, n: usize) -> f64 {
    format!("{x:.n$}").parse().unwrap()
}

/// numpy `np.round(x, n)`: rint(x * 10^n) / 10^n.
pub fn np_round_n(x: f64, n: i32) -> f64 {
    let s = 10f64.powi(n);
    (x * s).round_ties_even() / s
}

/// Python float repr for keys such as "unit=1.0".
pub fn py_repr(x: f64) -> String {
    format!("{x:?}")
}

/// Euclidean norm of a row the way numpy computes it along a contiguous axis.
pub fn norm_rows(x: &[f64]) -> f64 {
    let sq: Vec<f64> = x.iter().map(|v| v * v).collect();
    pairwise_sum(&sq).sqrt()
}

/// Sorted unique values (`np.unique`).
pub fn unique(a: &[f64]) -> Vec<f64> {
    let mut v = a.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    v.dedup();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy() {
        // np.percentile([1,2,3,4,10], 20) == 1.8 ; ([5, 1, 3], 15) == 1.6
        assert!((percentile(&[1.0, 2.0, 3.0, 4.0, 10.0], 20.0) - 1.8).abs() < 1e-12);
        assert!((percentile(&[5.0, 1.0, 3.0], 15.0) - 1.6).abs() < 1e-12);
        assert_eq!(percentile(&[7.0], 50.0), 7.0);
    }

    #[test]
    fn interp_basics() {
        let xp = [0.0, 1.0, 3.0];
        let fp = [0.0, 1.0, 2.0];
        assert_eq!(interp(2.0, &xp, &fp), 1.5);
        assert_eq!(interp(1.0, &xp, &fp), 1.0);
        assert_eq!(interp(5.0, &xp, &fp), 2.0);
    }

    #[test]
    fn rounding() {
        assert_eq!(round_i(2.5), 2);
        assert_eq!(round_i(3.5), 4);
        assert_eq!(round_n(0.125, 2), 0.12);
        assert_eq!(py_repr(1.0), "1.0");
    }
}
