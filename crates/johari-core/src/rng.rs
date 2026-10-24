//! Random draws.

use core::num::NonZero;

/// The finalizer of `SplitMix64`.
pub(crate) const fn mix(x: u64) -> u64 {
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// PRNG for a proposal.
pub(crate) struct Rng(u64);

impl Rng {
    /// The generator of the proposal after `pairs_shown` pairs.
    pub(crate) const fn new(pairs_shown: usize) -> Self {
        Self(pairs_shown as u64)
    }

    /// The next 64 random bits.
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix(self.0)
    }

    /// Uniformly sample an integer in [0, `n`).
    #[must_use]
    #[expect(clippy::cast_possible_truncation, reason = "the low word is taken on purpose")]
    fn below(&mut self, n: NonZero<u32>) -> u32 {
        let n = n.get();
        let mut m = (self.next_u64() >> 32) * u64::from(n);
        let mut low = m as u32;
        if low < n {
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                m = (self.next_u64() >> 32) * u64::from(n);
                low = m as u32;
            }
        }
        (m >> 32) as u32
    }

    /// Uniformly sample a truth value.
    #[must_use]
    pub(crate) fn bool(&mut self) -> bool {
        self.next_u64() >> 63 == 1
    }

    /// Uniformly sample an element of `items`. Returns `None` if `items` is empty.
    #[must_use]
    pub(crate) fn choice<T: Copy>(&mut self, items: &[T]) -> Option<T> {
        #[expect(clippy::cast_possible_truncation, reason = "slices here are far shorter than u32::MAX")]
        let len = NonZero::new(items.len() as u32)?;
        Some(items[self.below(len) as usize])
    }
}
