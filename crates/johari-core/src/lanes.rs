//! SIMD operations.

use core::simd::cmp::SimdPartialOrd;
use core::simd::{Select, f32x4};

/// This is a hack to prevent LLVM from turning static values into `v128.const`s. V8 rebuilds `v128.const`s, which is slower than
/// just loading them. `black_box` is overly broad for this purpose and blocks some optimizations that are still desirable.
#[cfg(target_arch = "wasm32")]
pub(crate) fn from_memory<T>(at: &'static &'static T) -> &'static T {
    use core::arch::asm;
    use core::ptr::from_ref as ptr_from_ref;

    let mut address = ptr_from_ref(at);
    #[expect(clippy::pointers_in_nomem_asm_block, reason = "the address is passed through without being read")]
    unsafe {
        asm!("/* {0} */", inout(local) address, options(pure, nomem, nostack, preserves_flags));
    }
    unsafe { *address }
}

#[cfg(not(target_arch = "wasm32"))]
#[expect(clippy::trivially_copy_pass_by_ref)]
pub(crate) fn from_memory<T>(at: &'static &'static T) -> &'static T {
    at
}

/// sqrt(x) per lane.
pub(crate) fn sqrt_lanes(x: f32x4) -> f32x4 {
    #[cfg(target_feature = "simd128")]
    {
        use core::arch::wasm32::f32x4_sqrt;
        f32x4_sqrt(x.into()).into()
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        use core::f32::math::sqrt;
        f32x4::from_array(x.to_array().map(sqrt))
    }
}

/// NaN-unaware min(a, b) per lane.
pub(crate) fn min_lanes<V: Copy + SimdPartialOrd<Mask: Select<V>>>(a: V, b: V) -> V {
    a.simd_lt(b).select(a, b)
}

/// NaN-unaware max(a, b) per lane.
pub(crate) fn max_lanes<V: Copy + SimdPartialOrd<Mask: Select<V>>>(a: V, b: V) -> V {
    a.simd_gt(b).select(a, b)
}

/// `n` rounded up to whole f32 lanes.
pub(crate) const fn padded(n: usize) -> usize {
    n.next_multiple_of(f32x4::LEN)
}
