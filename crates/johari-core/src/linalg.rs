//! Linear algebra routines.

use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::{max, min, minmax};
use core::f64::math::sqrt;
use core::hint::assert_unchecked;
use core::iter::zip;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut, Mul, Range, Sub};
use core::ptr::{from_mut as ptr_from_mut, from_ref as ptr_from_ref};
use core::simd::{Simd, SimdElement, f64x2};

/// A row-major matrix.
pub(crate) struct Matrix<T = f64> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Copy> Matrix<T> {
    /// The `rows × cols` zero matrix.
    pub(crate) fn zero(rows: usize, cols: usize) -> Self
    where
        T: Default,
    {
        Self { rows, cols, data: vec![T::default(); rows * cols] }
    }

    /// The number of rows.
    pub(crate) const fn rows(&self) -> usize {
        self.rows
    }

    /// Row `i`.
    pub(crate) fn row(&self, i: usize) -> &[T] {
        &self.data[self.row_range(i)]
    }

    /// Mutable access to row `i`.
    pub(crate) fn row_mut(&mut self, i: usize) -> &mut [T] {
        let range = self.row_range(i);
        &mut self.data[range]
    }

    /// Access to row `target` and row `source` so that one can be updated from the other. The rows must differ.
    pub(crate) fn two_rows(&mut self, target: usize, source: usize) -> (&mut [T], &[T]) {
        let ranges = [self.row_range(target), self.row_range(source)];
        let [target, source] = self.data.get_disjoint_mut(ranges).expect("two distinct rows");
        (target, source)
    }

    /// The range within `data` spanned by row `i`.
    fn row_range(&self, i: usize) -> Range<usize> {
        i * self.cols..(i + 1) * self.cols
    }

    /// Entry `(i, j)`.
    pub(crate) fn get(&self, i: usize, j: usize) -> T {
        self.data[i * self.cols + j]
    }

    /// Mutable reference to entry `(i, j)`.
    pub(crate) fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.data[i * self.cols + j]
    }
}

/// a·b.
#[inline(never)]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "lengths must be equal");
    let len = min(a.len(), b.len());
    let ((a_quads, a_rest), (b_quads, b_rest)) = (a[..len].as_chunks::<4>(), b[..len].as_chunks::<4>());
    let (mut low, mut high) = (f64x2::splat(0.), f64x2::splat(0.));
    for (x, y) in zip(a_quads, b_quads) {
        low += f64x2::from_array([x[0], x[1]]) * f64x2::from_array([y[0], y[1]]);
        high += f64x2::from_array([x[2], x[3]]) * f64x2::from_array([y[2], y[3]]);
    }
    let [left, right] = (low + high).to_array();
    zip(a_rest, b_rest).fold(left + right, |sum, (x, y)| sum + x * y)
}

/// x - a·y into x.
pub(crate) fn sub_scaled(x: &mut [f64], a: f64, y: &[f64]) {
    sub_scaled_in_lanes::<_, 2>(x, a, y);
}

/// x - a·y into x.
fn sub_scaled_in_lanes<T, const N: usize>(x: &mut [T], a: T, y: &[T])
where
    T: SimdElement + Sub<Output = T> + Mul<Output = T>,
    Simd<T, N>: Sub<Output = Simd<T, N>> + Mul<Output = Simd<T, N>>,
{
    debug_assert_eq!(x.len(), y.len(), "lengths must be equal");
    let ((x_lanes, x_rest), (y_lanes, y_rest)) = (x.as_chunks_mut(), y.as_chunks());
    let a_lanes = Simd::splat(a);
    for (x, &y) in zip(x_lanes, y_lanes) {
        *x = (Simd::from_array(*x) - a_lanes * Simd::from_array(y)).to_array();
    }
    for (x, &y) in zip(x_rest, y_rest) {
        *x = *x - a * y;
    }
}

/// The sparsity pattern of the lower Cholesky factor L of a symmetric positive definite n × n matrix A.
///
/// A is factored with its rows and columns permuted into a fill-reducing elimination order (see [`Self::ordered`]).
/// Call an index in the caller's original order a "coordinate" and an index in the elimination order a "row" or "column".
/// L and everything stored along the structure use rows, while [`Self::position`] takes coordinates.
/// A vector indexed by coordinate is "placed" when it is instead stored by row, with coordinate c's entry at index `row_of[c]`.
/// So, L·Lᵀ = P·A·Pᵀ, where P is the permutation matrix that places a vector.
///
/// A matrix stored along the structure is a flat slice of length n + nnz, where nnz is the number of entries below the diagonal.
/// The first n entries are on the main diagonal. Then, below the diagonal, column k holds the entries at positions n + t
/// for t in `starts[k]..starts[k + 1]`, entry t in row `rows[t]`.
pub(crate) struct Structure {
    /// Column k's rows below the diagonal are `rows[starts[k]..starts[k + 1]]`, ascending.
    /// Index t into `rows` is index n + t along the structure.
    starts: Vec<usize>,
    rows: Vec<usize>,
    /// The same entries indexed by row. Row i's entries left of the diagonal are `left[left_starts[i]..left_starts[i + 1]]`,
    /// each as [column k, index t into `rows`], by ascending column.
    left_starts: Vec<usize>,
    left: Vec<[usize; 2]>,
    /// For each column, the index into `rows` where its trailing run of consecutive rows begins.
    /// From the index to the end of the column, the rows are r, r + 1, r + 2, ...
    run: Vec<usize>,
    /// The elimination order. `coordinate[row]` is the coordinate at a row. `row_of[c]` is the row of coordinate c.
    coordinate: Vec<usize>,
    row_of: Vec<usize>,
}

impl Structure {
    /// The structure for symmetric n × n matrices whose last `dense` rows and columns may be entirely nonzero,
    /// and whose other entries off the diagonal are zero except at `entries`. Each entry is a pair of coordinates (row, column)
    /// in either triangle. Entries on the diagonal are ignored since the diagonal is always stored.
    pub(crate) fn ordered(n: usize, dense: usize, entries: impl IntoIterator<Item = [usize; 2]>) -> Self {
        let (coordinate, starts, mut rows) = min_degree(n, dense, entries);
        let mut row_of = vec![0; n];
        for (row, &c) in coordinate.iter().enumerate() {
            row_of[c] = row;
        }
        // Count by row and sum, then fill column by column, so each row's entries come by ascending column.
        let mut left_starts = vec![0; n + 1];
        for row in &mut rows {
            *row = row_of[*row];
            left_starts[*row + 1] += 1;
        }
        for i in 0..n {
            left_starts[i + 1] += left_starts[i];
        }
        let mut next = left_starts.clone();
        let mut left = vec![[0; 2]; rows.len()];
        for k in 0..n {
            for &row in &rows[starts[k]..starts[k + 1]] {
                left[next[row]][0] = k;
                next[row] += 1;
            }
        }
        // Refill the columns row by row so each column's rows come ascending.
        let mut next = starts.clone();
        for (i, entries) in left_starts.windows(2).enumerate() {
            for [k, t] in &mut left[entries[0]..entries[1]] {
                *t = next[*k];
                rows[*t] = i;
                next[*k] += 1;
            }
        }
        let mut run = Vec::with_capacity(n);
        for i in 0..n {
            let start = starts[i];
            let mut t = starts[i + 1];
            if t > start {
                t -= 1;
                while t > start && rows[t - 1] + 1 == rows[t] {
                    t -= 1;
                }
            }
            run.push(t);
        }
        Self { starts, rows, left_starts, left, run, coordinate, row_of }
    }

    /// The size n of the n × n matrices.
    fn order(&self) -> usize {
        self.starts.len() - 1
    }

    /// The length of a matrix stored along the structure.
    fn len(&self) -> usize {
        self.order() + self.rows.len()
    }

    /// The rows below the diagonal where column k may be nonzero.
    fn column(&self, k: usize) -> &[usize] {
        &self.rows[self.range(k)]
    }

    /// The indices into `rows` of column k's entries below the diagonal. Adding n gives their indices along the structure.
    fn range(&self, k: usize) -> Range<usize> {
        self.starts[k]..self.starts[k + 1]
    }

    /// Given column k's entries from index `from` into `rows` to the column's end,
    /// returns the entries before the column's run of consecutive rows, then the entries within the run.
    fn split_at_run(&self, k: usize, from: usize) -> [Range<usize>; 2] {
        let split = max(from, self.run[k]);
        [from..split, split..self.starts[k + 1]]
    }

    /// The index along the structure of entry (i, j) of a symmetric matrix, where i and j are coordinates.
    /// The entry must be on the diagonal, on the structure, or on its mirror.
    /// Entry (c, c) is at index `row_of[c]`, which is also where a placed vector holds coordinate c.
    pub(crate) fn position(&self, i: usize, j: usize) -> usize {
        let [column, row] = minmax(self.row_of[i], self.row_of[j]);
        if row == column {
            return row;
        }
        let offset = self.column(column).binary_search(&row).expect("the entry is on the structure");
        self.order() + self.starts[column] + offset
    }
}

/// Calls `each` with the index of every set bit in the bitset `words` in ascending order. Bit b of word w has index w·64 + b.
fn for_each_one(words: &[u64], mut each: impl FnMut(usize)) {
    for (w, &word) in words.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            each(w * 64 + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
}

/// The index of the lowest set bit in the bitset `words`, if any.
fn first_one(words: &[u64]) -> Option<usize> {
    words.iter().enumerate().find(|&(_, &word)| word != 0).map(|(w, &word)| w * 64 + word.trailing_zeros() as usize)
}

/// Sets bit `b` of the bitset `words`.
fn set(words: &mut [u64], b: usize) {
    words[b / 64] |= 1 << (b % 64);
}

/// Clears bit `b` of the bitset `words`.
fn clear(words: &mut [u64], b: usize) {
    words[b / 64] &= !(1 << (b % 64));
}

/// Computes a minimum degree elimination order for matrices described by [`Structure::ordered`].
/// Returns `(order, starts, rows)`, where `order[k]` is the k-th coordinate eliminated and column k's rows below the diagonal are
/// `rows[starts[k]..starts[k + 1]]`, given as coordinates (not rows) in an arbitrary order.
fn min_degree(n: usize, dense: usize, entries: impl IntoIterator<Item = [usize; 2]>) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let sparse = n.saturating_sub(dense);
    let words = sparse.div_ceil(64);
    // `adjacent[row(v)]` = neighbors of coordinate v not yet eliminated
    let mut adjacent = vec![0; sparse * words];
    // The words of row v.
    let row = |v| v * words..(v + 1) * words;
    // The index of bit b of row v in the entire table.
    let at = |v, b| 64 * words * v + b;

    for [i, j] in entries {
        if i != j && i < sparse && j < sparse {
            set(&mut adjacent, at(i, j));
            set(&mut adjacent, at(j, i));
        }
    }

    let degree = |adjacent: &[u64], v| -> u32 { adjacent[row(v)].iter().map(|word| word.count_ones()).sum() };
    // The degree of each coordinate. Eliminating v only changes its neighbors' degrees.
    let mut degrees: Vec<_> = (0..sparse).map(|v| degree(&adjacent, v)).collect();
    // `by_degree[row(d)]` = live coordinates of degree d
    let mut by_degree = vec![0; sparse * words];
    for (v, &d) in degrees.iter().enumerate() {
        set(&mut by_degree, at(d as usize, v));
    }

    let mut order = Vec::with_capacity(n);
    let mut starts = Vec::with_capacity(n + 1);
    let mut rows = Vec::new();
    starts.push(0);
    // The eliminated coordinate's neighbors. This is reused across eliminations.
    let mut neighbors = vec![0; words];
    for _ in 0..sparse {
        // The lowest index in the least degree's bitset.
        let v = by_degree.chunks_exact(words).find_map(first_one).expect("a coordinate left");
        clear(&mut by_degree, at(degrees[v] as usize, v));
        neighbors.copy_from_slice(&adjacent[row(v)]);
        // Each neighbor joins the others' clique and drops v.
        for_each_one(&neighbors, |u| {
            rows.push(u);
            for (target, &source) in zip(&mut adjacent[row(u)], &neighbors) {
                *target |= source;
            }
            clear(&mut adjacent, at(u, u));
            clear(&mut adjacent, at(u, v));
            clear(&mut by_degree, at(degrees[u] as usize, u));
            degrees[u] = degree(&adjacent, u);
            set(&mut by_degree, at(degrees[u] as usize, u));
        });
        rows.extend(sparse..n);
        order.push(v);
        starts.push(rows.len());
    }
    for k in sparse..n {
        rows.extend(k + 1..n);
        order.push(k);
        starts.push(rows.len());
    }
    (order, starts, rows)
}

/// Values in a structure's layout.
#[repr(transparent)]
pub(crate) struct View<K>(PhantomData<K>, [f64]);

// Slices take at most `isize::MAX` bytes.
const MAX_LEN: usize = isize::MAX.unsigned_abs() / size_of::<f64>();

impl<K> View<K> {
    /// `values` as a view of kind `K`.
    fn new(values: &[f64]) -> &Self {
        // SAFETY: `View` is `repr(transparent)` over `[f64]`.
        unsafe { &*(ptr_from_ref(values) as *const Self) }
    }

    /// `values` as a mutable view of kind `K`.
    fn new_mut(values: &mut [f64]) -> &mut Self {
        // SAFETY: `View` is `repr(transparent)` over `[f64]`.
        unsafe { &mut *(ptr_from_mut(values) as *mut Self) }
    }
}

impl<K> Deref for View<K> {
    type Target = [f64];

    fn deref(&self) -> &[f64] {
        // SAFETY: See `MAX_LEN`.
        unsafe { assert_unchecked(self.1.len() <= MAX_LEN) };
        &self.1
    }
}

impl<K> DerefMut for View<K> {
    fn deref_mut(&mut self) -> &mut [f64] {
        // SAFETY: See `MAX_LEN`.
        unsafe { assert_unchecked(self.1.len() <= MAX_LEN) };
        &mut self.1
    }
}

/// A lower Cholesky factor L stored along a [`Structure`].
struct Lower(Vec<f64>);

impl Deref for Lower {
    type Target = View<Self>;

    fn deref(&self) -> &View<Self> {
        View::new(&self.0)
    }
}

/// A symmetric matrix stored along a [`Structure`].
#[derive(Clone)]
pub(crate) struct Symmetric(Vec<f64>);

impl Symmetric {
    /// The zero matrix along `structure`.
    pub(crate) fn zero(structure: &Structure) -> Self {
        Self(vec![0.; structure.len()])
    }
}

impl Deref for Symmetric {
    type Target = View<Self>;

    fn deref(&self) -> &View<Self> {
        View::new(&self.0)
    }
}

impl DerefMut for Symmetric {
    fn deref_mut(&mut self) -> &mut View<Self> {
        View::new_mut(&mut self.0)
    }
}

/// A placed vector. Coordinate c's entry is at [`Structure::position`] `(c, c)`.
#[derive(Clone)]
pub(crate) struct Placed(Vec<f64>);

impl Placed {
    /// The zero vector over `structure`'s coordinates.
    pub(crate) fn zero(structure: &Structure) -> Self {
        Self(vec![0.; structure.order()])
    }
}

impl Deref for Placed {
    type Target = View<Self>;

    fn deref(&self) -> &View<Self> {
        View::new(&self.0)
    }
}

impl DerefMut for Placed {
    fn deref_mut(&mut self) -> &mut View<Self> {
        View::new_mut(&mut self.0)
    }
}

impl Structure {
    /// Converts a placed vector into a coordinate-indexed vector.
    fn unplace(&self, placed: &View<Placed>) -> Vec<f64> {
        self.row_of.iter().map(|&row| placed[row]).collect()
    }

    /// Overwrites `a`, the lower triangle of A, with the lower Cholesky factor L (L·Lᵀ = P·A·Pᵀ).
    /// Returns false if A is not finite or not positive definite, leaving `a` partly overwritten.
    fn factor(&self, a: &mut View<Symmetric>) -> bool {
        // This is a left-looking factorization. Column i of L is computed from A's column i
        // and the finished columns k < i with an entry in row i, taken in ascending order:
        //
        // L[i, i] = sqrt(A[i, i] - Σₖ L[i, k]²)
        // L[j, i] = (A[j, i] - Σₖ L[j, k]·L[i, k]) / L[i, i]    for each row j > i of column i
        let n = self.order();
        let (diag, below) = a.split_at_mut(n);
        let mut sums = vec![0.; n];
        for i in 0..n {
            for t in self.range(i) {
                sums[self.rows[t]] = below[t];
            }
            let mut pivot = diag[i];
            for &[k, t] in &self.left[self.left_starts[i]..self.left_starts[i + 1]] {
                let scale = below[t];
                pivot -= scale * scale;
                let [scattered, run] = self.split_at_run(k, t + 1);
                for s in scattered {
                    sums[self.rows[s]] -= scale * below[s];
                }
                if let Some(&first) = self.rows[run.clone()].first() {
                    for (sum, &l) in zip(&mut sums[first..first + run.len()], &below[run]) {
                        *sum -= scale * l;
                    }
                }
            }
            // Checking that each pivot is finite and positive is enough to determine whether L is finite.
            // Every entry L[i, k] is squared into row i's pivot, so a non-finite entry would make that pivot non-finite.
            if !(pivot > 0. && pivot < f64::INFINITY) {
                return false;
            }
            let d = sqrt(pivot);
            diag[i] = d;
            for t in self.range(i) {
                let row = self.rows[t];
                below[t] = sums[row] / d;
                sums[row] = 0.;
            }
        }
        true
    }

    /// Overwrites the placed vector `b` with (L·Lᵀ)⁻¹·b for the factor `l`.
    fn solve_placed(&self, l: &View<Lower>, b: &mut View<Placed>) {
        let (diag, below) = l.split_at(self.order());
        for (k, &d) in diag.iter().enumerate() {
            let y = b[k] / d;
            b[k] = y;
            for t in self.range(k) {
                b[self.rows[t]] -= below[t] * y;
            }
        }
        for (k, &d) in diag.iter().enumerate().rev() {
            let mut x = b[k];
            for t in self.range(k) {
                x -= below[t] * b[self.rows[t]];
            }
            b[k] = x / d;
        }
    }

    /// Overwrites B with Pᵀ·L⁻¹·P·B for the factor `l`: L⁻¹·P·B, with rows back in coordinate order.
    #[expect(clippy::cast_possible_truncation, reason = "the solves run in f32")]
    fn solve_lower(&self, l: &View<Lower>, b: &mut Matrix<f32>) {
        let (diag, below) = l.split_at(self.order());
        let at = &self.coordinate;
        for (k, &d) in diag.iter().enumerate() {
            let source = at[k];
            let inv = (1. / d) as f32;
            for x in b.row_mut(source) {
                *x *= inv;
            }
            for t in self.range(k) {
                let (target, x_k) = b.two_rows(at[self.rows[t]], source);
                sub_scaled_in_lanes::<_, 4>(target, below[t] as f32, x_k);
            }
        }
    }

    /// Overwrites B with Pᵀ·L⁻ᵀ·P·B for the factor `l`.
    #[expect(clippy::cast_possible_truncation, reason = "the solves run in f32")]
    fn solve_lower_transposed(&self, l: &View<Lower>, b: &mut Matrix<f32>) {
        let (diag, below) = l.split_at(self.order());
        let at = &self.coordinate;
        for (k, &d) in diag.iter().enumerate().rev() {
            let target = at[k];
            for t in self.range(k) {
                let (x_k, other) = b.two_rows(target, at[self.rows[t]]);
                sub_scaled_in_lanes::<_, 4>(x_k, below[t] as f32, other);
            }
            let inv = (1. / d) as f32;
            for x in b.row_mut(target) {
                *x *= inv;
            }
        }
    }

    /// Overwrites `z` with the selected inverse Z of L·Lᵀ for the factor `l`.
    ///
    /// L·Lᵀ = P·A·Pᵀ and Z = P·A⁻¹·Pᵀ, so entry (c, d) of A⁻¹ is at [`Self::position`] `(c, d)` for coordinates c and d.
    fn selected_inverse(&self, l: &View<Lower>, z: &mut View<Symmetric>) {
        // We use the Takahashi equations to do this efficiently, only reading entries on the structure.
        //
        // Z·L·Lᵀ = I, so Z·L = L⁻ᵀ, which is upper triangular with diagonal entries 1/L[k, k].
        // Let R be the set of rows below the diagonal where column k of L may be nonzero.
        // Entry (i, k) of Z·L is Z[i, k]·L[k, k] + Σ_{j ∈ R} Z[i, j]·L[j, k]. This is 0 for i > k and 1/L[k, k] for i = k, so:
        //
        // Z[i, k] = -(1/L[k, k])·Σ_{j ∈ R} Z[i, j]·L[j, k]                for i > k
        // Z[k, k] = 1/L[k, k]² - (1/L[k, k])·Σ_{j ∈ R} Z[k, j]·L[j, k]
        //
        // Of column k's rows past k, only those in R are on the structure, so the first line is computed just for them.
        // Then, we read Z[i, j] for i, j ∈ R. As both are past k, that entry is in a later column, which is already done.
        // We also know it's on the structure because eliminating k joined R into a clique (see `min_degree`), so for j < i in R,
        // row i is in column j. The second line reads the Z[k, j] = Z[j, k] that the first line just gave.
        let n = self.order();
        let (l_diag, l_below) = l.split_at(n);
        let (diag, below) = z.split_at_mut(n);
        // Each row i ∈ R's sum Σ_{j ∈ R} Z[i, j]·L[j, k], by place in R.
        let mut sums = vec![0.; n];

        for k in (0..n).rev() {
            let range = self.range(k);
            let rows = &self.rows[range.clone()];
            let column = &l_below[range.clone()];
            let sums = &mut sums[..rows.len()];
            sums.fill(0.);

            for (place, &j, &l_jk) in zip!(0.., rows, column) {
                // Z[j, j].
                let mut own = diag[j] * l_jk;
                // Column j's entries at R's rows past j.
                let later = &rows[place + 1..];
                if let Some(&first) = later.first() {
                    let (start, run, end) = (self.starts[j], self.run[j], self.starts[j + 1]);
                    let run_first = if run < end { self.rows[run] } else { usize::MAX };
                    if first >= run_first && range.start + place + 1 >= self.run[k] {
                        let entries = &below[run + first - run_first..][..later.len()];
                        for (sum, &z_ij, &l_ik) in zip!(&mut sums[place + 1..], entries, &column[place + 1..]) {
                            *sum += z_ij * l_jk;
                            own += z_ij * l_ik;
                        }
                    } else {
                        let mut at = start;
                        for (sum, &l_ik, &row) in zip!(&mut sums[place + 1..], &column[place + 1..], later) {
                            if row >= run_first {
                                at = run + row - run_first;
                            } else {
                                while self.rows[at] != row {
                                    at += 1;
                                }
                            }
                            let z_ij = below[at];
                            *sum += z_ij * l_jk;
                            own += z_ij * l_ik;
                        }
                    }
                }
                sums[place] += own;
            }

            let inv = 1. / l_diag[k];
            let mut own = 0.;
            for (entry, &sum, &l_jk) in zip!(&mut below[range], sums.iter(), column) {
                *entry = -inv * sum;
                own += *entry * l_jk;
            }
            diag[k] = inv * inv - inv * own;
        }
    }

    /// Return Z = (L·Lᵀ)⁻¹ for the factor `l`, as a coordinate-indexed dense matrix.
    ///
    /// Entry (c, d) of the result is Z[`row_of[c]`, `row_of[d]`].
    fn inverse(&self, l: &View<Lower>) -> Matrix {
        // We use the Takahashi equations (see `Self::selected_inverse`) for every row below each column k, not just the structure's.
        // By symmetry, column k below the diagonal is row k past it, which the first equation gives as a sum of rows past k:
        //
        // Z[k, i] = -Σ_{j ∈ R} (L[j, k]/L[k, k])·Z[j, i]    for i > k
        //
        // So, Z is built one row at a time in elimination order, where each such sum runs over contiguous entries. The sums also read
        // their rows left of the diagonal, so each finished row is copied into its column. Z is then put in coordinate order.
        let n = self.order();
        let (l_diag, l_below) = l.split_at(n);
        let mut z = Matrix::zero(n, n);
        for k in (0..n).rev() {
            let rows = self.column(k);
            let column = &l_below[self.range(k)];
            let inv = 1. / l_diag[k];
            for (&j, &l_jk) in zip(rows, column) {
                let (row_k, row_j) = z.two_rows(k, j);
                sub_scaled(&mut row_k[k + 1..], inv * l_jk, &row_j[k + 1..]);
            }
            let row = z.row_mut(k);
            let sum: f64 = zip(rows, column).map(|(&j, &l_jk)| row[j] * l_jk).sum();
            row[k] = inv * (inv - sum);
            let (done, later) = z.data.split_at_mut((k + 1) * n);
            for (&x, row) in zip(&done[k * n + k + 1..], later.chunks_exact_mut(n)) {
                row[k] = x;
            }
        }
        let row_of = &self.row_of;
        let mut data = Vec::with_capacity(n * n);
        for &i in row_of {
            let row = z.row(i);
            data.extend(row_of.iter().map(|&j| row[j]));
        }
        Matrix { rows: n, cols: n, data }
    }
}

/// A Cholesky factor L.
pub(crate) struct Cholesky {
    structure: Rc<Structure>,
    l: Lower,
}

impl Cholesky {
    /// The factor of A whose lower triangle `a` is stored along `structure`. Returns `None` if A is not finite or not positive definite.
    pub(crate) fn factor(structure: Rc<Structure>, mut a: Symmetric) -> Option<Self> {
        structure.factor(&mut a).then_some(Self { structure, l: Lower(a.0) })
    }

    /// The selected inverse of A.
    pub(crate) fn selected_inverse(&self) -> Symmetric {
        let mut z = Symmetric::zero(&self.structure);
        self.structure.selected_inverse(&self.l, &mut z);
        z
    }

    /// Overwrites the placed vector `b` with A⁻¹·b.
    pub(crate) fn solve_placed(&self, b: &mut View<Placed>) {
        self.structure.solve_placed(&self.l, b);
    }

    /// The placed vector `placed` in coordinate-indexed form.
    pub(crate) fn unplace(&self, placed: &View<Placed>) -> Vec<f64> {
        self.structure.unplace(placed)
    }

    /// Overwrites B with A⁻¹·B.
    pub(crate) fn solve_matrix(&self, b: &mut Matrix<f32>) {
        debug_assert_eq!(b.rows, self.structure.order(), "B has a row per coordinate");
        self.structure.solve_lower(&self.l, b);
        self.structure.solve_lower_transposed(&self.l, b);
    }

    /// The quadratic form bᵀ·A⁻¹·b of each column b of B.
    pub(crate) fn inverse_quadratic_forms(&self, mut b: Matrix<f32>) -> Vec<f32> {
        debug_assert_eq!(b.rows, self.structure.order(), "B has a row per coordinate");
        self.structure.solve_lower(&self.l, &mut b);
        let mut forms = vec![0.; b.cols];
        for k in 0..b.rows {
            for (form, y) in zip(&mut forms, b.row(k)) {
                *form += y * y;
            }
        }
        forms
    }

    /// The dense inverse A⁻¹.
    pub(crate) fn inverse(&self) -> Matrix {
        self.structure.inverse(&self.l)
    }
}

#[cfg(test)]
mod tests {
    use super::{Cholesky, Matrix, Structure, Symmetric};
    use crate::testing::Lcg;
    use alloc::rc::Rc;
    use alloc::vec;

    /// Factors the symmetric matrix whose lower triangle is `a`'s along `structure`. Returns `None` if it doesn't factor.
    fn factor(a: &Matrix, structure: Structure) -> Option<Cholesky> {
        let mut stored = Symmetric::zero(&structure);
        for i in 0..structure.order() {
            for j in 0..=i {
                stored[structure.position(i, j)] = a.get(i, j);
            }
        }
        Cholesky::factor(Rc::new(structure), stored)
    }

    #[test]
    fn min_degree_eliminates_low_degree_first() {
        let mut rng = Lcg(3);
        for trial in 0..40 {
            let n = 1 + trial % 23;
            let dense = (trial % 4).min(n);
            #[expect(clippy::cast_possible_truncation, reason = "test sizes are tiny")]
            let entries = (0..2 * n).map(|_| [rng.below(n as u64) as usize, rng.below(n as u64) as usize]);
            let structure = Structure::ordered(n, dense, entries);
            let mut seen = vec![false; n];
            for (row, &c) in structure.coordinate.iter().enumerate() {
                assert!(!seen[c] && structure.row_of[c] == row, "trial {trial}: {c} once, at row {row}");
                seen[c] = true;
            }
            assert!(
                seen.iter().all(|&s| s) && structure.coordinate[n - dense..].iter().copied().eq(n - dense..n),
                "trial {trial}: permutation; dense last"
            );
        }

        let hub = 3;
        let edges = (0..7).filter(|&v| v != hub).map(|v| [hub, v]);
        let structure = Structure::ordered(7, 0, edges);
        assert!(structure.row_of[hub] >= 5, "hub at row {}", structure.row_of[hub]);
        assert!((0..7).map(|k| structure.column(k).len()).sum::<usize>() == 6, "no fill beyond edges");
    }

    #[test]
    fn cholesky_factorization() {
        let n = 4;
        for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e200] {
            for i in 0..n {
                for j in 0..=i {
                    let mut a = Matrix::zero(n, n);
                    for r in 0..n {
                        for c in 0..=r {
                            *a.get_mut(r, c) = if r == c { 4. } else { 1. };
                        }
                    }
                    *a.get_mut(i, j) = bad;

                    let f = factor(&a, Structure::ordered(n, n, []));
                    let want = bad.is_finite() && i == j;
                    assert!(f.is_some() == want, "A[{i}, {j}] = {bad}: finite and positive definite A");
                    let finite = |factor: Cholesky| factor.l.iter().all(|x| x.is_finite());
                    assert!(f.is_none_or(finite), "A[{i}, {j}] = {bad}: finite factor");
                }
            }
        }
    }
}
