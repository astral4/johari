//! Utilities for working with a function's value and derivatives at a point.

use core::ops::{Add, Div, Mul, Sub};

/// A function f's value and first two derivatives at one point.
#[derive(Clone, Copy, Default)]
pub(crate) struct Jet<T> {
    pub(crate) value: T,
    pub(crate) d1: T,
    pub(crate) d2: T,
}

impl<T: Copy + Add<Output = T> + Mul<Output = T>> Jet<T> {
    /// The jet of a·f for a constant `a`.
    pub(crate) fn scaled(self, a: T) -> Self {
        Self { value: a * self.value, d1: a * self.d1, d2: a * self.d2 }
    }

    /// The jet of a·f + b for constants `a` and `b`.
    pub(crate) fn affine(self, a: T, b: T) -> Self {
        Self { value: a * self.value + b, d1: a * self.d1, d2: a * self.d2 }
    }
}

impl<T> Jet<T> {
    /// The value and derivatives through `each`.
    pub(crate) fn map<U>(self, mut each: impl FnMut(T) -> U) -> Jet<U> {
        Jet { value: each(self.value), d1: each(self.d1), d2: each(self.d2) }
    }
}

impl<T: Copy + Sub<Output = T> + Mul<Output = T> + Div<Output = T>> Jet<T> {
    /// The slopes of ln f for f > 0.
    ///
    /// ```text
    /// (ln f)′ = f′/f
    /// (ln f)″ = f″/f - (f′/f)²
    /// ```
    pub(crate) fn log_slopes(self) -> Slopes<T> {
        let (first, second) = (self.d1 / self.value, self.d2 / self.value);
        Slopes { d1: first, d2: second - first * first }
    }
}

impl<T: Add<Output = T>> Add for Jet<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self { value: self.value + other.value, d1: self.d1 + other.d1, d2: self.d2 + other.d2 }
    }
}

impl<T: Sub<Output = T>> Sub for Jet<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self { value: self.value - other.value, d1: self.d1 - other.d1, d2: self.d2 - other.d2 }
    }
}

/// A function f's first two derivatives at one point.
#[derive(Clone, Copy, Default)]
pub(crate) struct Slopes<T> {
    pub(crate) d1: T,
    pub(crate) d2: T,
}

impl Slopes<f64> {
    /// The slopes of a·f for a constant `a`.
    pub(crate) fn scaled(self, a: f64) -> Self {
        Self { d1: a * self.d1, d2: a * self.d2 }
    }

    /// Applies the chain rule. From f's slopes at g(x) (`self`) and g's slopes at x, the slopes of f∘g at x are:
    ///
    /// ```text
    /// (f∘g)′ = f′·g′
    /// (f∘g)″ = f′·g″ + f″·g′²
    /// ```
    pub(crate) fn chain(self, g: Self) -> Self {
        Self { d1: self.d1 * g.d1, d2: self.d1 * g.d2 + self.d2 * (g.d1 * g.d1) }
    }
}

impl<T: Add<Output = T>> Add for Slopes<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self { d1: self.d1 + other.d1, d2: self.d2 + other.d2 }
    }
}
