//! The final list order and tie groups.

use crate::posterior::Posterior;
use crate::rank_stats::PairTerms;
use alloc::vec;
use alloc::vec::Vec;
use core::iter::zip;

/// A session's tie cost τ; i.e. what a false tie costs in the final list's loss.
#[derive(Clone, Copy)]
pub(crate) struct TieCost(f64);

impl TieCost {
    #[must_use]
    pub(crate) const fn new(x: f64) -> Option<Self> {
        if x > 0. && x <= 1. { Some(Self(x)) } else { None }
    }

    pub(crate) const fn get(self) -> f64 {
        self.0
    }
}

/// A ranked list.
pub(crate) struct RankedList {
    /// Items, best first.
    pub(crate) order: Vec<u32>,
    /// Sizes of consecutive tie groups over `order`.
    pub(crate) group_sizes: Vec<u32>,
}

impl RankedList {
    /// The model's final list under the posterior at the given tie cost.
    pub(crate) fn new(posterior: &Posterior, tie_cost: TieCost) -> Self {
        let terms = PairTerms::new(posterior);
        let order = win_order(&terms);
        let group_sizes = tie_group_sizes(&terms, &order, tie_cost);
        Self { order, group_sizes }
    }
}

/// The list order computed via Copeland's method. Each pair prefers the ordering with the smaller expected weight of a reversal,
/// and the items go by how many of their pairs prefer them on top.
fn win_order(terms: &PairTerms) -> Vec<u32> {
    // L(order) = Σ_{k < m} B[order[m], order[k]] is, up to a constant, the expected loss of the list with nothing tied.
    // The cost of a pair listed one way is the weight of it being noticeably the other way.
    let n = terms.n_items();
    let mut wins = vec![0; n];
    for i in 0..n {
        for (j, &entry) in zip(i + 1.., &terms.row(i)[i + 1..]) {
            // [B[i, j], B[j, i]].
            let [ahead, behind, _] = entry.map(f64::from);
            let margin = terms.rounding(ahead + behind);
            let [u, v] = if ahead - behind > margin {
                [2, 0]
            } else if behind - ahead > margin {
                [0, 2]
            } else {
                [1, 1]
            };
            wins[i] += u;
            wins[j] += v;
        }
    }
    let top = 2 * n.saturating_sub(1);
    // `starts[top - w]` = where the items of count w go, by index
    let mut starts = vec![0; top + 2];
    for &w in &wins {
        starts[top - w + 1] += 1;
    }
    for k in 1..starts.len() {
        starts[k] += starts[k - 1];
    }
    let mut order = vec![0; n];
    for (item, &w) in zip(0.., &wins) {
        let slot = &mut starts[top - w];
        order[*slot] = item;
        *slot += 1;
    }
    order
}

/// Cut `order` into tie groups based on the minimum expected loss at the given tie cost. Returns the group sizes.
///
/// Listing u above l costs 1 if l is noticeably above u (a reversal) and 1 - τ if the two are within δ (a false order).
/// Tying them costs τ if either is noticeably above the other (a false tie).
/// So, in expectation, tying the pair instead of listing it in order changes the loss by
///
/// ```text
/// τ·(B[u, l] + B[l, u]) - B[l, u] - (1 - τ)·(W - B[u, l] - B[l, u])  =  B[u, l] - (1 - τ)·W
/// ```
///
/// which no other pair's claim enters, and the groups minimize
///
/// ```text
/// Σ_{tied k < m} (B[order[k], order[m]] - (1 - τ)·W[order[k], order[m]])
/// ```
///
/// exactly via dynamic programming over where the last group starts. A cost tie goes to the shorter last group.
/// At τ = 1, every term is B[u, l] ≥ 0, so nothing ties.
fn tie_group_sizes(terms: &PairTerms, order: &[u32], tie_cost: TieCost) -> Vec<u32> {
    let n = order.len();
    let false_order_cost = 1. - tie_cost.get();
    // best[e] = minimum cost of grouping order[..e]
    let mut best = vec![0.; n + 1];
    // start[e] = starting position of the last group in order[..e]
    let mut start = vec![0; n + 1];
    // row[k] = Σ_{k < m < e} (B - (1 - τ)·W); i.e. the cost of tying order[k] with the items after it up to e
    let mut row = vec![0.; n];
    for e in 1..=n {
        // [B[last, other], B[other, last], W] at each other item.
        let last = terms.row(order[e - 1] as usize);
        for (slot, &upper) in zip(&mut row, &order[..e - 1]) {
            let [_, upper_beyond, weight] = last[upper as usize].map(f64::from);
            *slot += upper_beyond - false_order_cost * weight;
        }
        best[e] = f64::INFINITY;
        let mut group = 0.;
        for s in (0..e).rev() {
            group += row[s];
            if best[s] + group < best[e] {
                (best[e], start[e]) = (best[s] + group, s);
            }
        }
    }
    let mut sizes = Vec::new();
    let mut e = n;
    while e > 0 {
        #[expect(clippy::cast_possible_truncation, reason = "group size is at most roster size")]
        sizes.push((e - start[e]) as u32);
        e = start[e];
    }
    sizes.reverse();
    sizes
}

#[cfg(test)]
mod tests {
    use super::{RankedList, TieCost, tie_group_sizes, win_order};
    use crate::observations::{Bucket, Observations, PairObs};
    use crate::posterior::fit;
    use crate::rank_stats::PairTerms;
    use crate::testing::{Lcg, TIE_COST};
    use alloc::vec;
    use alloc::vec::Vec;
    use core::cmp::{Ordering, Reverse};
    use core::f64::math::sqrt;
    use core::iter::{repeat_n, zip};

    /// The terms over `n` items whose pair i < j has the entry `terms(i, j)` = [B[i, j], B[j, i], W[i, j]].
    #[expect(clippy::cast_possible_truncation, reason = "the terms are kept in f32")]
    fn table(n: usize, mut terms: impl FnMut(usize, usize) -> [f64; 3]) -> PairTerms {
        PairTerms::from_fn(n, |i, j| terms(i, j).map(|x| x as f32))
    }

    /// [B[i, j], B[j, i], W[i, j]].
    fn entry(terms: &PairTerms, i: usize, j: usize) -> [f64; 3] {
        terms.row(i)[j].map(f64::from)
    }

    fn random_terms(n: usize, rng: &mut Lcg, mut ahead: impl FnMut(&mut Lcg, usize, usize) -> f64) -> PairTerms {
        table(n, |i, j| {
            let (weight, apart) = (rng.unit(), rng.unit());
            let ahead = ahead(rng, i, j);
            [weight * apart * ahead, weight * apart * (1. - ahead), weight]
        })
    }

    #[test]
    fn order_matches_ranking() {
        let mut rng = Lcg(2);
        for trial in 0..300 {
            let n = 1 + trial % 12;
            let class: Vec<_> = (0..n).map(|_| rng.below(4)).collect();
            let terms = random_terms(n, &mut rng, |rng, i, j| {
                let share = 0.55 + 0.45 * rng.unit();
                match class[i].cmp(&class[j]) {
                    Ordering::Greater => share,
                    Ordering::Less => 1. - share,
                    Ordering::Equal => 0.5,
                }
            });
            #[expect(clippy::cast_possible_truncation, reason = "test sizes are tiny")]
            let mut want: Vec<_> = (0..n as u32).collect();
            want.sort_by_key(|&i| (Reverse(class[i as usize]), i));
            assert_eq!(win_order(&terms), want, "trial {trial}");
        }
    }

    #[test]
    fn order_by_wins_then_index() {
        let mut rng = Lcg(7);
        for trial in 0..300 {
            let n = 1 + trial % 12;
            let terms = random_terms(n, &mut rng, |rng, _, _| if rng.below(4) == 0 { 0.5 } else { rng.unit() });
            let count = |i| -> f64 {
                (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let [ahead, behind, _] = entry(&terms, i, j);
                        let margin = terms.rounding(ahead + behind);
                        if ahead - behind > margin {
                            1.
                        } else if behind - ahead > margin {
                            0.
                        } else {
                            0.5
                        }
                    })
                    .sum()
            };
            let wins: Vec<_> = (0..n).map(count).collect();
            #[expect(clippy::cast_possible_truncation, reason = "test sizes are tiny")]
            let mut want: Vec<_> = (0..n as u32).collect();
            want.sort_by(|&a, &b| wins[b as usize].total_cmp(&wins[a as usize]).then(a.cmp(&b)));
            assert_eq!(win_order(&terms), want, "trial {trial}");
        }
    }

    #[test]
    fn pairs_prefer_only_beyond_rounding() {
        for (part, want) in [(1e-7, [0, 1]), (1e-3, [1, 0])] {
            let terms = table(2, |_, _| [0.3, 0.3 * (1. + part), 0.8]);
            assert_eq!(win_order(&terms), want, "apart by {part}");
        }
    }

    #[test]
    fn tie_groups_minimize_expected_loss() {
        let mut rng = Lcg(3);
        for trial in 0..300 {
            let n = 1 + trial % 8;
            let terms = random_terms(n, &mut rng, |rng, _, _| rng.unit());
            let mut order: Vec<_> = (0..).take(n).collect();
            rng.shuffle(&mut order);
            let tie_cost = 1. - rng.unit();
            // The cost of the grouping that puts position k in group[k].
            let cost = |group: &[usize]| {
                let mut total = 0.;
                for k in 0..n {
                    for m in k + 1..n {
                        if group[k] == group[m] {
                            let [forward, _, weight] = entry(&terms, order[k] as usize, order[m] as usize);
                            total += forward - (1. - tie_cost) * weight;
                        }
                    }
                }
                total
            };
            // Bit j of `cuts` ends a group after position j.
            let least = (0..1u32 << (n - 1))
                .map(|cuts| cost(&(0..n).map(|k| (cuts & ((1 << k) - 1)).count_ones() as usize).collect::<Vec<_>>()))
                .fold(f64::INFINITY, f64::min);
            let sizes = tie_group_sizes(&terms, &order, TieCost::new(tie_cost).expect("in (0, 1]"));
            let group: Vec<_> = zip(0.., &sizes).flat_map(|(g, &size)| repeat_n(g, size as usize)).collect();
            assert_eq!(group.len(), n, "the groups cover the order");
            assert!((cost(&group) - least).abs() <= 1e-12, "trial {trial}: {} vs the least {least}", cost(&group));
        }
    }

    #[test]
    fn tie_cost_one() {
        // τ = 1 should result in no ties.
        let terms = table(3, |i, _| if i == 0 { [0., 0., 0.5] } else { [0.25, 0., 0.5] });
        assert_eq!(tie_group_sizes(&terms, &[0, 1, 2], TieCost::new(1.).unwrap()), [1, 1, 1], "a tie cost of 1");
    }

    #[test]
    fn rounding_reordering() {
        let obs = Observations {
            buckets: vec![Bucket::Fine; 4],
            pairs: vec![PairObs::Pick { above: 0, below: 1 }, PairObs::Pick { above: 2, below: 3 }],
        };
        let mut posterior = fit(&obs);
        let mu = posterior.mu.clone();
        for tip in [-1e-11, 1e-11] {
            for item in 0..4 {
                let sd = sqrt(posterior.covariance.get(item, item));
                posterior.mu[item] = mu[item] + if item >= 2 { tip * sd } else { 0. };
            }
            let list = RankedList::new(&posterior, TIE_COST);
            assert_eq!(list.order, [0, 2, 1, 3], "tipped by {tip}");
        }
    }
}
