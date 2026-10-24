//! Gauss-Hermite quadrature.

/// A quadrature rule. Σₖ `weights[k]`·f(`nodes[k]`) approximates an integral of f.
pub(crate) struct Rule<const K: usize> {
    pub(crate) nodes: [f64; K],
    pub(crate) weights: [f64; K],
}

/// The probabilist's Hermite polynomial Heₙ(x) and its derivative.
///
/// ```text
/// He₋₁     = 0
/// He₀      = 1
/// He_{k+1} = x·Heₖ - k·He_{k-1}
///
/// He'ₙ     = n·He_{n-1}
/// ```
#[expect(clippy::cast_precision_loss, reason = "degrees are tiny")]
const fn hermite(n: usize, x: f64) -> (f64, f64) {
    let mut prev = 0.;
    let mut cur = 1.;
    let mut k = 0;
    while k < n {
        let next = x * cur - (k as f64) * prev;
        prev = cur;
        cur = next;
        k += 1;
    }
    (cur, (n as f64) * prev)
}

/// The `K` roots of the degree-`K` Hermite polynomial in ascending order.
#[expect(clippy::cast_precision_loss, reason = "grid sizes are tiny")]
const fn roots<const K: usize>() -> [f64; K] {
    let bound = K as f64 + 1.;
    let steps = 128 * K;
    let step = bound / (steps as f64);

    let mut nodes = [0.; K];
    let mut found = K / 2;
    let mut x_prev = 0.;
    let mut f_prev = hermite(K, x_prev).0;
    let mut i = 0;
    while i < steps {
        let x_next = x_prev + step;
        let f_next = hermite(K, x_next).0;
        if f_prev == 0. {
            nodes[found] = x_prev;
            found += 1;
        } else if (f_prev < 0. && f_next > 0.) || (f_prev > 0. && f_next < 0.) {
            // Bisection until no float lies strictly between lo and hi.
            let mut lo = x_prev;
            let mut hi = x_next;
            let mut f_lo = f_prev;
            nodes[found] = loop {
                let mid = f64::midpoint(lo, hi);
                if mid <= lo || mid >= hi {
                    break mid;
                }
                let f_mid = hermite(K, mid).0;
                if (f_mid < 0.) == (f_lo < 0.) {
                    lo = mid;
                    f_lo = f_mid;
                } else {
                    hi = mid;
                }
            };
            found += 1;
        }
        x_prev = x_next;
        f_prev = f_next;
        i += 1;
    }
    assert!(found == K, "the scan must bracket every root");
    let mut k = 0;
    while k < K / 2 {
        nodes[k] = -nodes[K - 1 - k];
        k += 1;
    }
    nodes
}

/// The Gauss-Hermite rule with `K` nodes for the weight e^(-x²/2). Its weights sum to 1, so Σₖ wₖ·f(m + s·xₖ) ≈ E[f(X)] for X ~ N(m, s²).
pub(crate) const fn gauss_hermite<const K: usize>() -> Rule<K> {
    assert!(K >= 1, "a rule needs a node");
    let nodes = roots();
    let mut weights = [0.; K];
    let mut total = 0.;
    let mut k = 0;
    while k < K {
        let deriv = hermite(K, nodes[k]).1;
        // wₖ ∝ 1 / He'_K(xₖ)²; the constant cancels in the normalization.
        weights[k] = 1. / (deriv * deriv);
        total += weights[k];
        k += 1;
    }
    let mut k = 0;
    while k < K {
        weights[k] /= total;
        k += 1;
    }
    Rule { nodes, weights }
}

#[cfg(test)]
mod tests {
    use super::{Rule, gauss_hermite, hermite};
    use core::iter::zip;

    /// Σₖ wₖ and Σₖ wₖ·xₖ².
    fn moments<const K: usize>(Rule { nodes, weights }: &Rule<K>) -> (f64, f64) {
        let mut total = 0.;
        let mut second = 0.;
        for (x, w) in zip(nodes, weights) {
            total += w;
            second += w * x * x;
        }
        (total, second)
    }

    #[test]
    fn hermite_rule() {
        let rule = gauss_hermite::<7>();
        let nodes = rule.nodes;
        assert!(nodes[3].to_bits() == 0, "the middle node is zero");
        for &x in &nodes {
            assert!(hermite(7, x).0.abs() < 1e-12, "every node is a zero of He₇");
        }
        let (total, second) = moments(&rule);
        assert!((total - 1.).abs() < 4. * f64::EPSILON, "the weights sum to 1");
        assert!((second - 1.).abs() < 1e-14, "the rule integrates x² against the standard normal");
    }
}
