//! Logic for determining indistinguishable items from a log.

use crate::observations::{N_BUCKETS, Observations, PairObs};
use crate::rng::mix;
use alloc::vec;
use alloc::vec::Vec;
use core::iter::chain;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Won = 0,
    Lost = 1,
    Same = 2,
}

fn answer_hash(other: u32, side: Side) -> u64 {
    mix((u64::from(other) << 2) | side as u64)
}

/// Classes of twins of a log.
pub(crate) struct Twins {
    /// The root of each class in ascending order.
    roots: Vec<usize>,
    /// The class of each item.
    class: Vec<usize>,
    /// The size of each class.
    sizes: Vec<f64>,
    /// The last item of each class. This is a twin of the class' root unless the class only has its root.
    lasts: Vec<usize>,
}

/// Not yet in a class, in [`Twins::of`].
const UNCLASSED: usize = usize::MAX;

impl Twins {
    /// The classes of twins in `obs`.
    ///
    /// Twins are items in the same bucket with the same answer against every other item, and between them
    /// either no answer (false twins) or "same" (true twins). Swapping two twins maps the log to itself,
    /// so the posterior is symmetric in them and the model can't tell them apart.
    pub(crate) fn of(obs: &Observations) -> Self {
        let n = obs.buckets.len();

        // Item i's answers are `answers[at[i]..at[i + 1]]`.
        let mut at = vec![0; n + 1];
        for &pair in &obs.pairs {
            for item in pair.indices() {
                at[item] += 1;
            }
        }
        for k in 1..=n {
            at[k] += at[k - 1];
        }
        let mut answers = vec![(0, Side::Same); at[n]];
        let mut signature = vec![0u64; n];
        for &pair in &obs.pairs {
            let ([a, b], sides) = match pair {
                PairObs::Pick { above, below } => ([above, below], [Side::Won, Side::Lost]),
                PairObs::Same(pair) => (pair.items(), [Side::Same; 2]),
            };
            for (item, other, side) in [(a, b, sides[0]), (b, a, sides[1])] {
                at[item as usize] -= 1;
                answers[at[item as usize]] = (other, side);
                signature[item as usize] = signature[item as usize].wrapping_add(answer_hash(other, side));
            }
        }

        let row = |item| &answers[at[item as usize]..at[item as usize + 1]];
        let closed_signature = |item| signature[item as usize].wrapping_add(answer_hash(item, Side::Same));

        let mut tally = vec![[0; 3]; n];
        // Whether `a` and `b` are twins. Checks for true twins if `closed` and false twins otherwise.
        let mut twins = |a, b, closed| {
            let mut between = [0; 3];
            for &(other, side) in row(a) {
                if other == b {
                    between[side as usize] += 1;
                } else {
                    tally[other as usize][side as usize] += 1;
                }
            }
            for &(other, side) in row(b).iter().filter(|&&(other, _)| other != a) {
                tally[other as usize][side as usize] -= 1;
            }
            let mut alike = obs.buckets[a as usize] == obs.buckets[b as usize];
            for &(other, _) in chain(row(a), row(b)).filter(|&&(other, _)| other != a && other != b) {
                alike &= tally[other as usize] == [0; 3];
                tally[other as usize] = [0; 3];
            }
            between == if closed { [0, 0, 1] } else { [0; 3] } && alike
        };

        let mut roots = Vec::new();
        let mut class = vec![UNCLASSED; n];
        let open = |roots: &mut Vec<_>, a| {
            roots.push(a as usize);
            roots.len() - 1
        };
        let mut untouched = [None; N_BUCKETS];
        #[expect(clippy::cast_possible_truncation, reason = "roster sizes are far below u32::MAX")]
        for a in 0..n as u32 {
            if class[a as usize] != UNCLASSED {
                continue;
            }
            let mine = row(a);
            let Some(&(least, _)) = mine.iter().min_by_key(|&&(other, _)| row(other).len()) else {
                class[a as usize] = *untouched[obs.buckets[a as usize] as usize].get_or_insert_with(|| open(&mut roots, a));
                continue;
            };
            class[a as usize] = open(&mut roots, a);
            let false_twins = row(least).iter().map(|&(b, _)| (b, false));
            let true_twins = mine.iter().filter(|&&(_, side)| side == Side::Same).map(|&(b, _)| (b, true));
            for (b, closed) in chain(false_twins, true_twins) {
                let alike =
                    if closed { closed_signature(a) == closed_signature(b) } else { signature[a as usize] == signature[b as usize] };
                if class[b as usize] == UNCLASSED && alike && twins(a, b, closed) {
                    class[b as usize] = class[a as usize];
                }
            }
        }

        let mut sizes = vec![0.; roots.len()];
        let mut lasts = vec![0; roots.len()];
        for (item, &c) in class.iter().enumerate() {
            sizes[c] += 1.;
            lasts[c] = item;
        }
        Self { roots, class, sizes, lasts }
    }

    /// The root of each class in ascending order.
    pub(crate) fn roots(&self) -> &[usize] {
        &self.roots
    }

    /// The class of each item.
    pub(crate) fn class(&self) -> &[usize] {
        &self.class
    }

    /// The size of each class.
    pub(crate) fn sizes(&self) -> &[f64] {
        &self.sizes
    }

    /// The last item of each class. This is a twin of the class' root unless the class only has its root.
    pub(crate) fn lasts(&self) -> &[usize] {
        &self.lasts
    }

    /// The root of each item.
    pub(crate) fn item_roots(&self) -> impl Iterator<Item = usize> + '_ {
        self.class.iter().map(|&class| self.roots[class])
    }
}

#[cfg(test)]
mod tests {
    use super::Twins;
    use crate::observations::{Bucket, Observations, PairObs};
    use crate::testing::twin_log;
    use alloc::vec;

    #[test]
    fn untouched_items_within_bucket() {
        use Bucket::{Fine, Meh};
        let buckets = vec![Meh, Fine, Fine, Meh, Fine, Fine, Meh, Fine];
        let twins = Twins::of(&Observations { pairs: vec![PairObs::Pick { above: 1, below: 3 }], buckets });
        assert_eq!(twins.roots(), [0, 1, 2, 3]);
        assert_eq!(twins.class(), [0, 1, 2, 3, 2, 2, 0, 2]);
    }

    #[test]
    fn twin_kinds() {
        let twins = Twins::of(&twin_log());
        assert_eq!(twins.roots(), [0, 1, 3, 5, 6, 7, 8]);
        assert_eq!(twins.class(), [0, 1, 1, 2, 2, 3, 4, 5, 6, 6]);
    }
}
