#![allow(dead_code)]
use auto_ops::*;

use std::cmp::{Eq, Ord, PartialEq, PartialOrd};
use std::fmt;
use std::iter::{FromIterator, IntoIterator, Sum};
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::matrix::Matrix;

/// Vector orientation (column or row).
#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub enum Orientation {
    Column,
    Row,
}

impl Orientation {
    /// Transpose this orientation (column to row, row to column).
    pub fn transposed(&self) -> Self {
        match self {
            Orientation::Column => Orientation::Row,
            Orientation::Row => Orientation::Column,
        }
    }
}

/// Simple $m$-length vector.
#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct Vector {
    orientation: Orientation,
    elements: Vec<f64>,
}

/// Column vector constructor helper macro.
#[rustfmt::skip]
#[macro_export]
macro_rules! cvec {
    ($($a:expr),*) => { Vector::new(Orientation::Column, &vec![$($a as f64),*]) };
}

/// Row vector constructor helper macro.
#[rustfmt::skip]
#[macro_export]
macro_rules! rvec {
    ($($a:expr),*) => { Vector::new(Orientation::Row, &vec![$($a as f64),*]) };
}

impl Vector {
    pub fn new(orientation: Orientation, elements: &[f64]) -> Self {
        Self {
            orientation,
            elements: elements.to_vec(),
        }
    }

    pub fn col_from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        Self {
            orientation: Orientation::Column,
            elements: iter.into_iter().collect(),
        }
    }

    pub fn row_from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        Self {
            orientation: Orientation::Row,
            elements: iter.into_iter().collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn iter(&self) -> std::slice::Iter<f64> {
        self.elements.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<f64> {
        self.elements.iter_mut()
    }

    fn is_mul_compatible(&self, other: &Vector) -> bool {
        self.orientation == Orientation::Row
            && other.orientation == Orientation::Column
            && self.len() == other.len()
    }

    pub fn transposed(&self) -> Self {
        Self {
            orientation: self.orientation.transposed(),
            elements: self.elements.clone(),
        }
    }

    pub fn dot(&self, other: &Self) -> Option<f64> {
        (self.orientation == other.orientation).then_some(
            self.iter()
                .zip(other.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>(),
        )
    }

    /// Is this vector orthogonal to another.
    ///
    /// Returns `true` if $a$ (this vector) is orthogonal to $b$ (another) vector
    /// (that is if $a\cdot b<\epsilon<0$), or `false` if non-orthogonal or if this and/or the
    /// other are not vectors.
    ///
    /// # Examples
    /// ```
    /// use hw03::prelude::*;
    /// let a = cvec![1.0, 0.0, 0.0];
    /// let b = cvec![0.0, 2.0, 3.0];
    /// assert!(a.is_orthogonal(&b));
    /// ```
    ///
    /// ```should_panic
    /// use hw03::prelude::*;
    /// let a = cvec![1.0, 0.0, 0.0];
    /// let b = rvec![0.0, 2.0, 3.0];
    /// assert!(a.is_orthogonal(&b));
    /// ```
    ///
    /// ```should_panic
    /// use hw03::prelude::*;
    /// let a = ident![3];
    /// let b = rvec![0.0, 2.0, 3.0];
    /// assert!(a.is_orthogonal(&b));
    /// ```
    pub fn is_orthogonal(&self, other: &Self) -> bool {
        self.dot(other).map(|a| a <= f64::EPSILON).unwrap_or(false)
    }

    pub fn mag2(&self) -> f64 {
        self.iter().map(|a| a * a).sum::<f64>()
    }

    pub fn mag(&self) -> f64 {
        self.mag2().sqrt()
    }

    pub fn normed(&self) -> Self {
        if (1.0 - self.mag2()).abs() < f64::EPSILON {
            // for a = <a1,a2,...>
            // if a1^2+a2^2+...=1, then a^1/2 = a and a already normal
            self.clone()
        } else {
            let m = self.mag();
            self.iter().copied().map(|a| a / m).collect()
        }
    }

    pub fn scalar_proj(&self, other: &Self) -> Option<f64> {
        self.dot(other).map(|d| d / other.mag())
    }

    pub fn proj(&self, other: &Self) -> Option<Self> {
        let other = other.normed();
        self.scalar_proj(&other).map(|d| other * d)
    }
}

impl FromIterator<f64> for Vector {
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        Self::col_from_iter(iter)
    }
}

// Vector and scalar addition
impl_op_ex_commutative!(+|lhs: &Vector, rhs: &f64| -> Vector {
    Vector {
        elements: lhs.iter().map(|a| (*a) + rhs).collect(),
        ..*lhs
    }
});

// Vector and scalar subtraction
impl_op_ex!(-|lhs: &Vector, rhs: &f64| -> Vector { lhs + (-rhs) });

// Vector and scalar multiplication
impl_op_ex_commutative!(*|lhs: &Vector, rhs: &f64| -> Vector {
    Vector {
        elements: lhs.iter().map(|a| (*a) * rhs).collect(),
        ..*lhs
    }
});

// Vector and scalar division
impl_op_ex!(/|lhs: &Vector, rhs: &f64| -> Vector { lhs * (1.0 / rhs) });

impl_op_ex!(-|v: &Vector| -> Vector { v.clone() * (-1.0) });

impl_op_ex!(+|lhs: &Vector, rhs: &Vector| -> Vector {
    if lhs.orientation != rhs.orientation {
        panic!("Cannot add column and row vectors");
    }
    Vector {
        elements: lhs.elements
            .iter()
            .zip(rhs.elements.iter())
            .map(|(a, b)| a + b).collect(),
        orientation: lhs.orientation,
    }
});

impl_op_ex!(-|lhs: &Vector, rhs: &Vector| -> Vector {
    if lhs.orientation != rhs.orientation {
        panic!("Cannot subtract column and row vectors");
    }
    lhs + (-rhs)
});

//impl_op_ex!(*|lhs

impl Mul<Matrix> for Vector {
    type Output = Option<Self>;

    fn mul(self, other: Matrix) -> Self::Output {
        use std::iter::repeat;

        let m = self.len();
        if matches!(
            (self.orientation, other.shape()),
            (Orientation::Row, &(a, b)) if a == m && b == m
        ) {
            Some(Self::row_from_iter(
                repeat(self.iter())
                    .zip(other.col_iter())
                    .map(|(vi, ci)| vi.zip(ci).map(|(a, b)| a * b).sum()),
            ))
        } else {
            None
        }
    }
}

impl Sum for Vector {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        if let Some(mut res) = iter.next() {
            while let Some(next) = iter.next() {
                res = res + next;
            }
            res
        } else {
            Vector::new(Orientation::Column, &[0.0])
        }
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        let mut i = self.elements.iter();
        if let Some(a) = i.next() {
            write!(f, "{:.2}", a)?;
            while let Some(a) = i.next() {
                write!(f, " {:.2}", a)?;
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}
