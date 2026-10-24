//! Symbol reference:
//!
//! ```text
//! φ                           the fitted vector: θ, then c₁, log(c₂ - c₁), log δ
//! θ          theta            utilities, one per active item
//! εₚ, εₛ     careless         careless rates of picks and "same" answers
//! ηₚ, ηₛ     log_ratios       ηₖ = ln(εₖ/(1 - εₚ - εₛ)), each rate's log-ratio to careful answering
//! c₁, c₂     c1, c2           bucket cutpoints, fitted as c₁ and log(c₂ - c₁)
//! δ          delta            just-noticeable difference, fitted as log δ
//! σ₀         sigma0           prior sd of θ
//! β          beta_pair        pair sharpness
//! s_b        s_b              bucket noise scale
//! λ          lambda           ln s_b
//! Φ(x)                        the standard normal CDF
//! φ(x)                        the standard normal density
//! q, μ, Λ    posterior        the posterior approximation N(μ, Λ⁻¹)
//! Σ          covariance       q's covariance Λ⁻¹, or its block over θ
//! h          shift            a Gaussian's shift Λ·m; it and the precision Λ are its natural parameters
//! A          jacobian         an observation's row Jacobian ∂(x, y)/∂φ
//! T          site             a site's precision over its row's (x, y)
//! g          gap              a gap θᵢ - θⱼ
//! R                           an item's 0-based true rank; 0 is best
//! w(k)                        the position weight of a pair whose better item has rank k
//! W[i, j]    weights          E[w(min(Rᵢ, Rⱼ))], a pair's expected position weight
//! B[i, j]    beyond           E[w(min(Rᵢ, Rⱼ))·1{θᵢ - θⱼ > δ}], the part of W[i, j] in which i is noticeably above j
//! τ          tie_cost         the cost of a false tie in the final list
//! ```

#![no_std]
#![cfg_attr(target_arch = "wasm32", feature(asm_experimental_arch))]
#![feature(cmp_minmax)]
#![feature(core_float_math)]
#![feature(more_float_constants)]
#![feature(optimize_attribute)]
#![feature(portable_simd)]

extern crate alloc;

macro_rules! zip {
    ($a:expr, $b:expr, $c:expr $(,)?) => {
        ::core::iter::zip($a, ::core::iter::zip($b, $c)).map(|(a, (b, c))| (a, b, c))
    };
    ($a:expr, $b:expr, $c:expr, $d:expr $(,)?) => {
        ::core::iter::zip($a, ::core::iter::zip($b, ::core::iter::zip($c, $d))).map(|(a, (b, (c, d)))| (a, b, c, d))
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr $(,)?) => {
        ::core::iter::zip($a, ::core::iter::zip($b, ::core::iter::zip($c, ::core::iter::zip($d, $e))))
            .map(|(a, (b, (c, (d, e))))| (a, b, c, d, e))
    };
}

mod gaussian;
mod jet;
mod lanes;
mod linalg;
mod model;
mod observations;
mod posterior;
mod quadrature;
mod rank_stats;
mod rng;
mod special;
mod twins;

#[cfg(test)]
mod testing;
