//! Observation types over session-local item indices.

use alloc::vec;
use alloc::vec::Vec;

/// A bucket from the bucket stage in increasing order of utility.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bucket {
    Meh,
    Fine,
    Love,
}

impl Bucket {
    pub(crate) const ALL: [Self; 3] = [Self::Meh, Self::Fine, Self::Love];
}

const _: () = {
    let mut k = 0;
    while k < N_BUCKETS {
        assert!(Bucket::ALL[k] as usize == k, "`Bucket::ALL` is in the order of the discriminants");
        k += 1;
    }
};

pub(crate) const N_BUCKETS: usize = Bucket::ALL.len();

/// An unordered pair of distinct items.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Pair([u32; 2]);

impl Pair {
    /// Create a pair from two distinct items.
    pub(crate) const fn new(a: u32, b: u32) -> Self {
        debug_assert!(a != b, "a pair needs two distinct items");
        if a < b { Self([a, b]) } else { Self([b, a]) }
    }

    /// The items as `[lo, hi]`.
    pub(crate) const fn items(self) -> [u32; 2] {
        self.0
    }

    /// The items as `[lo, hi]` for indexing.
    pub(crate) const fn indices(self) -> [usize; 2] {
        [self.0[0] as usize, self.0[1] as usize]
    }
}

/// The number of distinct pairs over `n` items; i.e. C(n, 2).
pub(crate) const fn n_pairs(n: usize) -> usize {
    n * n.saturating_sub(1) / 2
}

/// A set of unordered item pairs.
pub(crate) struct PairSet {
    n: usize,
    bits: Vec<u64>,
}

impl PairSet {
    /// The empty set over `n` items.
    pub(crate) fn new(n: usize) -> Self {
        Self { n, bits: vec![0; (n * n.saturating_sub(1)).div_ceil(64)] }
    }

    fn index(&self, pair: Pair) -> usize {
        let [lo, hi] = pair.indices();
        lo * self.n + hi
    }

    /// Whether the pair is in the set.
    pub(crate) fn contains(&self, pair: Pair) -> bool {
        let at = self.index(pair);
        (self.bits[at / 64] >> (at % 64)) & 1 == 1
    }

    /// Add a pair.
    pub(crate) fn insert(&mut self, pair: Pair) {
        let at = self.index(pair);
        self.bits[at / 64] |= 1 << (at % 64);
    }
}

/// An answered pair.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PairObs {
    /// `above` was picked over `below`.
    Pick { above: u32, below: u32 },
    /// About the same.
    Same(Pair),
}

impl PairObs {
    /// Whether the answer was a pick or tie.
    pub(crate) const fn kind(self) -> AnswerKind {
        match self {
            Self::Pick { .. } => AnswerKind::Pick,
            Self::Same(_) => AnswerKind::Same,
        }
    }

    /// The pair items as `[above, below]` for a pick or `[lo, hi]` for a tie, for indexing.
    pub(crate) const fn indices(self) -> [usize; 2] {
        match self {
            Self::Pick { above, below } => [above as usize, below as usize],
            Self::Same(pair) => pair.indices(),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum AnswerKind {
    Pick,
    Same,
}

impl AnswerKind {
    pub(crate) const ALL: [Self; 2] = [Self::Pick, Self::Same];
}

const _: () = {
    let mut k = 0;
    while k < AnswerKind::ALL.len() {
        assert!(AnswerKind::ALL[k] as usize == k, "`AnswerKind::ALL` is in the order of the discriminants");
        k += 1;
    }
};

/// The observation log that the posterior is fitted to.
#[derive(PartialEq, Eq)]
pub(crate) struct Observations {
    /// The bucket of each item by local index.
    pub(crate) buckets: Vec<Bucket>,
    /// Pairs in the order they were answered.
    pub(crate) pairs: Vec<PairObs>,
}
