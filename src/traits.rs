use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Zero value of type.
pub trait Zero {
    const ZERO: Self;
}

/// One value of type.
pub trait One {
    const ONE: Self;
}

/// Epsilon of type.
pub trait Epsilon {
    const EPSILON: Self;
}

/// Value of type to a power.
pub trait Pow {
    fn pow<N: Into<f64> + Copy>(&self, n: N) -> Self;

    fn squared(&self) -> Self
    where
        Self: Sized,
    {
        self.pow(2)
    }
}

/// Root of a value of type.
///
/// Note, this is incomplete: only returns the first nth root.
pub trait Root {
    fn root(&self, n: i32) -> Self;

    fn sqrt(&self) -> Self
    where
        Self: Sized,
    {
        self.root(2)
    }
}

/// Conjugate of value of type.
///
/// Commonly, the conjugate of a real value is itself, while the conjugate of a complex
/// value $a+bi$ (its "complex conjugate") is $a-bi$.
pub trait Conjugate {
    type Output;

    fn conj(&self) -> Self::Output;
}

/// Monolithic type trait alias for vector and matrix data
/// (satisfied by either real, i.e. `f64`, or complex values).
pub trait Primitive = Neg<Output = Self>
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Sum<Self>
    + PartialEq<Self>
    + Zero
    + One
    + Conjugate<Output = Self>
    + Pow
    + Root
    + Copy;

/// Random value of type.
pub trait Random {
    fn random() -> Self;
}

// Primitive implementation for `f64`
impl Zero for f64 {
    const ZERO: f64 = 0.0;
}

impl One for f64 {
    const ONE: f64 = 1.0;
}

impl Pow for f64 {
    fn pow<N: Into<f64>>(&self, n: N) -> Self {
        f64::powf(*self, n.into())
    }
}

impl Root for f64 {
    fn root(&self, n: i32) -> f64 {
        (*self).powf(1.0 / n as f64)
    }
}

impl Conjugate for f64 {
    type Output = f64;

    fn conj(&self) -> f64 {
        *self
    }
}

impl Random for f64 {
    fn random() -> f64 {
        rand::random()
    }
}

#[cfg(test)]
mod f64_tests {
    use crate::prelude::*;

    #[test]
    fn f64_trait_sqrt() {
        assert_eq!((289.0).sqrt(), 17.0);
    }

    #[test]
    fn f64_trait_conj() {
        assert_eq!((255.0).conj(), 255.0);
    }
}
