use auto_ops::*;
use std::iter::FromIterator;
use std::{fmt, fmt::Display};

use crate::matrix::{Matrix, MatrixShape};

macro_rules! impl_vector {
    ($this:tt, $other:tt) => {
        #[derive(Clone, PartialEq, PartialOrd, Debug)]
        pub struct $this {
            elements: Vec<f64>,
        }

        impl $this {
            /// New vector.
            pub fn new(elements: &[f64]) -> Self {
                Self {
                    elements: elements.to_vec(),
                }
            }

            pub fn from_iterator<I>(iter: I) -> Self
            where
                I: Iterator<Item = f64>,
            {
                let elements: Vec<f64> = iter.collect();
                Self { elements }
            }

            pub fn random(size: usize) -> Self {
                Self {
                    elements: (0..size).map(|_| rand::random::<f64>()).collect()
                }
            }

            /// Number of elements (i.e. vector length).
            pub fn len(&self) -> usize {
                self.elements.len()
            }

            /// Iterator over the vector elements.
            pub fn iter(&self) -> impl Iterator<Item = &f64> + Clone {
                self.elements.iter()
            }

            /// Mutable iterator over the vector elements.
            pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64> {
                self.elements.iter_mut()
            }

            /// Transpose this vector (column to row, row to column).
            pub fn t(&self) -> $other {
                $other {
                    elements: self.elements.clone(),
                }
            }

            fn hadamard_iter<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = f64> + 'a {
                self.iter().zip(other.iter()).map(|(a, b)| a * b)
            }

            /// Hadamard product of two vectors.
            pub fn hadamard(&self, other: &Self) -> Self {
                self.hadamard_iter(other).collect()
            }

            /// Dot product of two vectors.
            pub fn dot(&self, other: &Self) -> f64 {
                self.hadamard_iter(other).sum()
            }

            /// Is this vector orthogonal to another.
            pub fn is_orthogonal_to(&self, other: &Self) -> bool {
                self.dot(other) <= f64::EPSILON
            }

            /// Vector magnitude squared (2-norm).
            pub fn mag2(&self) -> f64 {
                self.iter().map(|a| a * a).sum()
            }

            /// Vector magnitude (2-norm).
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

            pub fn scalar_proj(&self, other: &Self) -> f64 {
                self.dot(other) / other.dot(other)
            }

            pub fn proj(&self, other: &Self) -> Self {
                other * self.scalar_proj(other)
            }
        }

        impl FromIterator<f64> for $this {
            fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
                Self {
                    elements: iter.into_iter().collect(),
                }
            }
        }

        // Vector and scalar addition
        impl_op_ex_commutative!(+ |lhs: &$this, rhs: &f64| -> $this {
            lhs.iter().copied().map(|a| a + rhs).collect()
        });

        // Vector and scalar subtraction
        impl_op_ex!(- |lhs: &$this, rhs: &f64| -> $this {
            lhs + (-rhs)
        });

        // Vector and scalar multiplication
        impl_op_ex_commutative!(* |lhs: &$this, rhs: &f64| -> $this {
            lhs.iter().copied().map(|a| a * rhs).collect()
        });

        // Vector and scalar division
        impl_op_ex!(/ |lhs: &$this, rhs: &f64| -> $this {
            lhs * (1.0 / rhs)
        });

        macro_rules! op_assign {
            ($op:tt) => {
                impl_op_ex!($op |lhs: &mut $this, rhs: &f64| {
                    lhs.iter_mut().for_each(|a| *a $op rhs);
                });
            }
        }

        // Vector and scalar operation assignment
        op_assign!(+=);
        op_assign!(-=);
        op_assign!(*=);
        op_assign!(/=);

        // Vector negation
        impl_op_ex!(-|v: &$this| -> $this {
            v.clone() * (-1.0)
        });

        // Vector and vector addition
        impl_op_ex!(+|lhs: &$this, rhs: &$this| -> $this {
            lhs.iter().zip(rhs.iter()).map(|(a, b)| a + b).collect()
        });

        // Vector and vector subtraction
        impl_op_ex!(-|lhs: &$this, rhs: &$this| -> $this {
            lhs + (-rhs)
        });

        macro_rules! op_assign {
            ($op:tt) => {
                impl_op_ex!($op |lhs: &mut $this, rhs: &$this| {
                    lhs.iter_mut().zip(rhs.iter()).for_each(|(a, b)| *a $op b);
                });
            }
        }

        // Vector and vector operation assignment
        op_assign!(+=);
        op_assign!(-=);

        impl Display for $this {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(
                    f,
                    "[{}]",
                    self.iter()
                        .map(|a| format!("{:.3}", a))
                        .collect::<Vec<String>>()
                        .join(" ")
                )?;
                Ok(())
            }
        }
    };
}

impl_vector!(ColVector, RowVector);
impl_vector!(RowVector, ColVector);

/// Column vector constructor helper macro.
#[macro_export]
macro_rules! cvec {
    ($($a:expr),*) => { ColVector::new(&vec![$($a as f64),*]) };
}

/// Row vector constructor helper macro.
#[macro_export]
macro_rules! rvec {
    ($($a:expr),*) => { RowVector::new(&vec![$($a as f64),*]) };
}

// Row vector and matrix multiplication.
impl_op_ex!(*|lhs: &RowVector, rhs: &Matrix| -> RowVector {
    use std::iter::repeat;

    if lhs.len() != rhs.shape().m() {
        panic!("Row vector and matrix multiplication size mismatch");
    }
    repeat(lhs.iter())
        .zip(rhs.col_iter())
        .map(|(vi, ci)| vi.zip(ci).map(|(a, b)| a * b).sum())
        .collect()
});

// Matrix and column vector multiplication.
impl_op_ex!(*|lhs: &Matrix, rhs: &ColVector| -> ColVector {
    use std::iter::repeat;

    if lhs.shape().n() != rhs.len() {
        panic!("Matrix and column vector multiplication size mismatch");
    }
    lhs.row_iter()
        .zip(repeat(rhs.iter()))
        .map(|(ri, vi)| ri.zip(vi).map(|(a, b)| a * b).sum())
        .collect()
});
