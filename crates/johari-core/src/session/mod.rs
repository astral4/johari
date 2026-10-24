//! Session state.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::OnceCell;
use core::iter::zip;
use core::mem::swap;

use crate::observations::{Observations, Pair, PairObs, PairSet, n_pairs};
use crate::posterior::{Posterior, fit};
use crate::rank_stats::rank_intervals;
use crate::rng::Rng;
use crate::selection::Scorer;

pub use crate::observations::Bucket;
pub use crate::rank_stats::Level;
pub use crate::ranking::{RankedList, TieCost};

#[cfg(feature = "bench")]
pub mod bench;

/// The answer to a shown pair.
#[derive(Clone, Copy)]
pub enum Outcome {
    /// Picked the first item shown.
    First,
    /// Picked the second item shown.
    Second,
    /// About the same.
    Same,
}

/// The kind of log entry retracted by [`Session::undo`].
#[derive(Clone, Copy)]
pub enum Undone {
    /// The log was empty.
    Nothing,
    /// A bucket mark.
    Mark,
    /// A shown pair.
    Pair,
}

/// A rejected session action.
#[derive(Clone, Copy)]
pub enum SessionError {
    /// An item is outside the roster.
    ItemOutOfRange,
    /// The item already has a mark.
    AlreadyMarked,
    /// A mark was attempted after a pair had been shown.
    BucketStageClosed,
    /// A pair includes the same item twice.
    PairNotDistinct,
    /// A pair includes an item without a bucket.
    ItemNotActive,
    /// The pair has already been shown this session.
    PairAlreadyShown,
}

/// A session's roster size.
#[derive(Clone, Copy)]
pub struct RosterSize(u32);

impl RosterSize {
    pub const MAX: u32 = 1 << 11;

    #[must_use]
    pub const fn new(n: u32) -> Option<Self> {
        const { assert!(Self::MAX.checked_mul(Self::MAX - 1).is_some(), "roster pairs are counted and indexed in 32 bits") };
        if n <= Self::MAX { Some(Self(n)) } else { None }
    }

    const fn get(self) -> usize {
        self.0 as usize
    }
}

/// The next pair to ask.
pub struct Proposal {
    /// The pair as roster items in display order.
    pub shown: [u32; 2],
    /// The answer's expected reduction in expected loss of the final list.
    pub value: f64,
}

/// The final list.
pub struct Review {
    /// The final list of roster items.
    pub list: RankedList,
    /// Lower end (inclusive) of the 0-based rank interval, parallel to the list's order.
    pub rank_lo: Vec<u32>,
    /// Upper end (inclusive) of the 0-based rank interval, parallel to the list's order.
    pub rank_hi: Vec<u32>,
}

/// The mark of a roster item.
#[derive(Clone, Copy)]
enum Mark {
    Unmarked,
    DontKnow,
    Bucket(Bucket),
}

impl Mark {
    /// The item's bucket.
    const fn bucket(self) -> Option<Bucket> {
        if let Self::Bucket(bucket) = self { Some(bucket) } else { None }
    }
}

/// The response for a pair in the pairs stage.
struct PairRecord {
    /// The pair as local indices in display order.
    shown: [u32; 2],
    /// The response, or `None` for a skip.
    outcome: Option<Outcome>,
}

impl PairRecord {
    /// The unordered pair.
    const fn pair(&self) -> Pair {
        Pair::new(self.shown[0], self.shown[1])
    }

    /// The record as a model observation.
    fn observation(&self) -> Option<PairObs> {
        let outcome = self.outcome?;
        let [a, b] = self.shown;
        Some(match outcome {
            Outcome::First => PairObs::Pick { above: a, below: b },
            Outcome::Second => PairObs::Pick { above: b, below: a },
            Outcome::Same => PairObs::Same(self.pair()),
        })
    }
}

/// A fit.
struct Fit {
    obs: Observations,
    tie_cost: TieCost,
    posterior: Posterior,
    scorer: OnceCell<Scorer>,
}

impl Fit {
    fn new(obs: Observations, tie_cost: TieCost) -> Self {
        Self { posterior: fit(&obs), obs, tie_cost, scorer: OnceCell::new() }
    }

    /// The proposal's values.
    fn scorer(&self) -> &Scorer {
        self.scorer.get_or_init(|| Scorer::new(&self.posterior, self.tie_cost))
    }
}

/// Session state.
pub struct Session {
    /// The mark of each roster item.
    marks: Vec<Mark>,
    /// The marked items in the order they were marked.
    marked: Vec<u32>,
    /// The pair log in the order shown.
    pairs: Vec<PairRecord>,
    /// The session's tie cost.
    tie_cost: TieCost,
    /// The last two fits, later first.
    fits: [Option<Box<Fit>>; 2],
}

impl Session {
    /// A new session over a roster of `roster_size` items using the given tie cost.
    #[must_use]
    pub fn new(roster_size: RosterSize, tie_cost: TieCost) -> Self {
        Self { marks: vec![Mark::Unmarked; roster_size.get()], marked: Vec::new(), pairs: Vec::new(), tie_cost, fits: [None, None] }
    }

    /// Record a bucket (or `None` for "don't know") for a roster item.
    ///
    /// # Errors
    ///
    /// Returns an error if a pair has been shown, the item index is out of bounds for the roster, or the item has already been marked.
    pub fn mark(&mut self, item: u32, bucket: Option<Bucket>) -> Result<(), SessionError> {
        if !self.pairs.is_empty() {
            return Err(SessionError::BucketStageClosed);
        }
        let Some(slot) = self.marks.get_mut(item as usize) else { return Err(SessionError::ItemOutOfRange) };
        if !matches!(slot, Mark::Unmarked) {
            return Err(SessionError::AlreadyMarked);
        }
        *slot = bucket.map_or(Mark::DontKnow, Mark::Bucket);
        self.marked.push(item);
        Ok(())
    }

    /// Append one shown pair with its answer (`None` for a skip).
    ///
    /// # Errors
    ///
    /// Returns an error if an item index is out of bounds for the roster, the pair repeats an item, an item has no bucket,
    /// or the pair has already been shown.
    pub fn record(&mut self, shown: [u32; 2], outcome: Option<Outcome>) -> Result<(), SessionError> {
        let [a, b] = shown;
        let (Some(mark_a), Some(mark_b)) = (self.marks.get(a as usize), self.marks.get(b as usize)) else {
            return Err(SessionError::ItemOutOfRange);
        };
        if a == b {
            return Err(SessionError::PairNotDistinct);
        }
        if mark_a.bucket().is_none() || mark_b.bucket().is_none() {
            return Err(SessionError::ItemNotActive);
        }
        let shown = shown.map(|item| self.local_of(item));
        let pair = Pair::new(shown[0], shown[1]);
        if self.pairs.iter().any(|r| r.pair() == pair) {
            return Err(SessionError::PairAlreadyShown);
        }
        self.pairs.push(PairRecord { shown, outcome });
        Ok(())
    }

    /// Retract the most recent data action.
    pub fn undo(&mut self) -> Undone {
        if self.pairs.pop().is_some() {
            Undone::Pair
        } else if let Some(item) = self.marked.pop() {
            self.marks[item as usize] = Mark::Unmarked;
            Undone::Mark
        } else {
            Undone::Nothing
        }
    }

    /// The pairs shown so far over `n_active` items. Returns `None` if all pairs of active items have been shown.
    fn shown_pairs(&self, n_active: usize) -> Option<PairSet> {
        (self.pairs.len() < n_pairs(n_active)).then(|| {
            let mut shown = PairSet::new(n_active);
            for r in &self.pairs {
                shown.insert(r.pair());
            }
            shown
        })
    }

    /// Choose the next pair from the posterior of the current log. Returns `None` if all pairs of active items have been shown.
    #[must_use]
    pub fn propose(&mut self) -> Option<Proposal> {
        let active = self.active();
        let seen = self.shown_pairs(active.len())?;
        let mut rng = Rng::new(self.pairs.len());
        let swap = rng.bool();
        let fit = self.fit();
        let (pair, value) = fit.scorer().propose(&fit.posterior, &seen, &mut rng);
        let [lo, hi] = pair.items();
        let shown = if swap { [hi, lo] } else { [lo, hi] }.map(|i| active[i as usize]);
        Some(Proposal { shown, value })
    }

    /// The final list from the posterior of the current log, with rank intervals at level `level`.
    #[must_use]
    pub fn review(&mut self, level: Level) -> Review {
        let active = self.active();
        let Fit { posterior, tie_cost, .. } = self.fit();
        let mut list = RankedList::new(posterior, *tie_cost);
        let intervals = rank_intervals(posterior, level);
        let (rank_lo, rank_hi) = list.order.iter().map(|&i| intervals[i as usize].into()).unzip();
        for item in &mut list.order {
            *item = active[*item as usize];
        }
        Review { list, rank_lo, rank_hi }
    }

    /// The local index of a bucketed roster item; i.e. the number of bucketed items before it.
    #[expect(clippy::cast_possible_truncation, reason = "roster sizes are far below u32::MAX")]
    fn local_of(&self, item: u32) -> u32 {
        self.marks[..item as usize].iter().filter(|mark| mark.bucket().is_some()).count() as u32
    }

    /// The bucketed roster items in ascending order.
    fn active(&self) -> Vec<u32> {
        zip(0.., &self.marks).filter_map(|(item, mark)| mark.bucket().is_some().then_some(item)).collect()
    }

    /// Bucket marks in local order, then answered pairs.
    fn observations(&self) -> Observations {
        let mut pairs = Vec::with_capacity(self.pairs.len());
        pairs.extend(self.pairs.iter().filter_map(PairRecord::observation));
        let buckets = self.marks.iter().filter_map(|mark| mark.bucket()).collect();
        Observations { buckets, pairs }
    }

    /// The fit of the current observations.
    #[inline(never)]
    fn fit(&mut self) -> &Fit {
        let obs = self.observations();
        let [last, second_last] = &mut self.fits;
        if last.as_ref().is_none_or(|fit| fit.obs != obs) {
            swap(last, second_last);
            if last.as_ref().is_none_or(|fit| fit.obs != obs) {
                // The earlier fit is dropped before the new one is made so that at most two are held at once.
                drop_fit(last.take());
                *last = Some(Box::new(Fit::new(obs, self.tie_cost)));
            }
        }
        last.as_ref().unwrap()
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        for fit in &mut self.fits {
            drop_fit(fit.take());
        }
    }
}

#[inline(never)]
fn drop_fit(fit: Option<Box<Fit>>) {
    drop(fit);
}

#[cfg(test)]
mod tests {
    use super::{Bucket, Level, Outcome, RosterSize, Session, SessionError, TieCost, Undone};
    use crate::testing::{Lcg, TIE_COST};
    use core::iter::chain;

    const LEVEL: Level = Level::new(0.8).unwrap();
    const OUTCOMES: [Option<Outcome>; 4] = [None, Some(Outcome::First), Some(Outcome::Second), Some(Outcome::Same)];

    /// A logged action.
    #[derive(Clone, Copy)]
    enum Action {
        Mark(u32, Option<Bucket>),
        Record([u32; 2], Option<Outcome>),
    }

    /// A session over a roster of `roster` items at the given tie cost, after the actions of `log`.
    fn replayed(roster: u32, tie_cost: TieCost, log: impl IntoIterator<Item = Action>) -> Session {
        let mut s = Session::new(RosterSize::new(roster).expect("a test roster"), tie_cost);
        for action in log {
            match action {
                Action::Mark(item, bucket) => s.mark(item, bucket).ok().expect("mark"),
                Action::Record(shown, outcome) => s.record(shown, outcome).ok().expect("record"),
            }
        }
        s
    }

    #[test]
    fn marks_index_the_roster() {
        let marks = [(7, Some(Bucket::Love)), (2, None), (5, Some(Bucket::Meh)), (0, Some(Bucket::Fine)), (8, None)];
        let mut s = replayed(9, TIE_COST, marks.map(|(item, bucket)| Action::Mark(item, bucket)));
        let listed = |s: &mut Session| {
            let mut order = s.review(LEVEL).list.order;
            order.sort_unstable();
            order
        };
        assert_eq!(listed(&mut s), [0, 5, 7]);
        assert!(matches!(s.mark(2, Some(Bucket::Fine)), Err(SessionError::AlreadyMarked)));
        assert!(matches!(s.record([7, 2], Some(Outcome::Same)), Err(SessionError::ItemNotActive)));
        assert!(s.record([7, 0], Some(Outcome::Same)).is_ok() && listed(&mut s) == [0, 5, 7]);
        assert!(matches!(s.undo(), Undone::Pair) && listed(&mut s) == [0, 5, 7]);
        assert!(matches!(s.undo(), Undone::Mark) && listed(&mut s) == [0, 5, 7]);
        assert!(matches!(s.undo(), Undone::Mark) && listed(&mut s) == [5, 7]);
        assert!(matches!(s.mark(2, None), Err(SessionError::AlreadyMarked)));
        assert!(s.mark(0, None).is_ok() && s.mark(8, Some(Bucket::Meh)).is_ok());
    }

    #[test]
    fn reject_records_in_order() {
        let marks = [(0, Some(Bucket::Fine)), (1, Some(Bucket::Love)), (2, None)];
        let mut s = replayed(4, TIE_COST, marks.map(|(item, bucket)| Action::Mark(item, bucket)));
        let record = |s: &mut Session, pair| s.record(pair, Some(Outcome::First));
        for pair in [[4, 0], [0, 4], [4, 4]] {
            assert!(matches!(record(&mut s, pair), Err(SessionError::ItemOutOfRange)), "{pair:?}: outside the roster");
        }
        assert!(matches!(record(&mut s, [2, 2]), Err(SessionError::PairNotDistinct)));
        for pair in [[0, 2], [3, 0]] {
            assert!(matches!(record(&mut s, pair), Err(SessionError::ItemNotActive)), "{pair:?}: don't know or unmarked");
        }
        assert!(record(&mut s, [1, 0]).is_ok());
        for pair in [[1, 0], [0, 1]] {
            assert!(matches!(record(&mut s, pair), Err(SessionError::PairAlreadyShown)), "{pair:?}: shown in either order");
        }
        assert!(matches!(s.mark(3, Some(Bucket::Meh)), Err(SessionError::BucketStageClosed)));
    }

    #[test]
    fn tiny_rosters() {
        for (roster, marks) in [(0, 0), (3, 0), (3, 1)] {
            let mut s = replayed(roster, TIE_COST, (0..marks).map(|item| Action::Mark(item, Some(Bucket::Fine))));
            assert!(s.propose().is_none(), "{marks} active: nothing to ask");
            let review = s.review(LEVEL);
            let listed = review.list.order.len() == marks as usize && review.list.group_sizes.len() == marks as usize;
            assert!(listed && review.rank_lo.len() == marks as usize, "{marks} active: listed");
        }
    }

    #[test]
    fn list_alike_items_by_index() {
        let marks = (0..4).map(|item| Action::Mark(item, Some(Bucket::Fine)));
        let same = [[2, 3], [0, 1]].map(|pair| Action::Record(pair, Some(Outcome::Same)));
        let order = replayed(4, TIE_COST, chain(marks, same)).review(LEVEL).list.order;
        assert_eq!(order, [0, 1, 2, 3]);
    }

    #[test]
    fn list_twins_by_index() {
        let marks = (0..12).map(|item| Action::Mark(item, Some(Bucket::Fine)));
        let answers = (1..5).map(|item| Action::Record([0, item], Some(Outcome::First)));
        let mut s = replayed(12, TIE_COST, marks.chain(answers).chain([Action::Record([3, 4], Some(Outcome::Same))]));
        let mut r = Lcg(4);
        for a in 5..12 {
            for b in a + 1..12 {
                // Every answer but the skip.
                let outcome = r.pick(&OUTCOMES[1..]);
                s.record([a, b], outcome).ok().expect("record");
                let order = s.review(LEVEL).list.order;
                let place = |item| order.iter().position(|&listed| listed == item);
                assert!(place(1) < place(2) && place(3) < place(4), "after answering ({a}, {b}): {order:?}");
            }
        }
    }
}
