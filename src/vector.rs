#![allow(unused_imports)]
use auto_ops::{impl_op_ex, impl_op_ex_commutative};

use std::cmp::PartialOrd;
use std::iter::{FromIterator, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::{fmt, fmt::Display};

use crate::matrix::{Matrix, Shape};
use crate::traits::{Conjugate, Epsilon, Pow, Primitive, Random, Root};

pub trait Vector {
    type ElementType: Primitive;

    /// New vector.
    fn new(elements: Vec<Self::ElementType>) -> Self
    where
        Self: Sized;

    fn elements(&self) -> &Vec<Self::ElementType>;

    fn elements_mut(&mut self) -> &mut Vec<Self::ElementType>;

    /// Number of elements (i.e. vector length).
    fn len(&self) -> usize {
        self.elements().len()
    }

    /// Iterator over the vector elements.
    fn iter(&self) -> std::slice::Iter<Self::ElementType> {
        self.elements().iter()
    }

    /// Mutable iterator over the vector elements.
    fn iter_mut(&mut self) -> std::slice::IterMut<Self::ElementType> {
        self.elements_mut().iter_mut()
    }

    /// Vector magnitude squared (2-norm).
    fn mag2(&self) -> Self::ElementType {
        self.iter().map(|&a| a * a).sum()
    }

    /// Vector magnitude (2-norm).
    fn mag(&self) -> Self::ElementType {
        self.mag2().sqrt()
    }

    /// Vector normalized.
    fn normed(&self) -> Self
    where
        Self: Sized,
    {
        let m = self.mag();
        Self::new(self.iter().copied().map(|a| a / m).collect())
    }
}

/// Hadamard iterator of vectors (i.e. element-wise product).
pub fn hadamard_iter<'l, A, B, U, V>(
    lhs: &'l U,
    rhs: &'l V,
) -> impl Iterator<Item = <A as Mul<B>>::Output> + 'l
where
    A: Primitive + Mul<B> + 'l,
    B: Primitive + 'l,
    U: Vector<ElementType = A>,
    V: Vector<ElementType = B>,
{
    lhs.iter().zip(rhs.iter()).map(|(&a, &b)| a * b)
}

/// Inner product of vectors.
pub trait InnerProduct<A, B, V>
where
    A: Primitive + Mul<B>,
    B: Primitive,
    V: Vector<ElementType = B>,
    <A as Mul<B>>::Output: Primitive,
    Self: Vector<ElementType = A> + Sized,
{
    fn dot(&self, other: &V) -> <A as Mul<B>>::Output {
        hadamard_iter(self, other).sum()
    }

    fn orthogonal(&self, other: &V) -> bool {
        //self.dot(other) < <A as Mul<B>>::Output::EPSILON
        self.dot(other) == <A as Mul<B>>::Output::ZERO
    }

    fn hadamard(&self, other: &Self) -> Self {
        Self::new(hadamard_iter(self, other).collect())
    }
}

/// Projection of vectors.
pub trait Project<T>
where
    T: Primitive,
    Self: Vector<ElementType = T> + InnerProduct<T, T, Self> + Sized,
{
    fn scalar_proj(&self, other: &Self) -> T {
        self.dot(other) / other.dot(other)
    }

    fn proj(&self, other: &Self) -> Self {
        let r = self.scalar_proj(other);
        Self::new(self.iter().map(|&a| a * r).collect())
    }
}

/// Column vector.
#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct ColVector<T: Primitive> {
    elements: Vec<T>,
}

/// Row vector.
#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct RowVector<T: Primitive> {
    elements: Vec<T>,
}

#[macro_export]
macro_rules! rcvec {
    ($($a:expr),*) => {
        ColVector::<f64>::new(vec![$(($a).into()),*])
    };
}

#[macro_export]
macro_rules! ccvec {
    ($($a:expr),*) => {
        ColVector::<Complex>::new(vec![$(($a).into()),*])
    };
}

#[macro_export]
macro_rules! rrvec {
    ($($a:expr),*) => {
        RowVector::<f64>::new(vec![$(($a).into()),*])
    };
}

#[macro_export]
macro_rules! crvec {
    ($($a:expr),*) => {
        RowVector::<Complex>::new(vec![$(($a).into()),*])
    };
}

// Vector and primitive operation implementation helper
macro_rules! impl_primitive_op {
    (@variant $this:ty, ($name:ident, $f:ident, $nameasn:ident, $fasn:ident), $lhs:ty, $rhs:ty) => {
        impl<T: Primitive> $name<$rhs> for $lhs {
            type Output = $this;

            fn $f(self, rhs: $rhs) -> $this {
                let mut res = self.clone();
                $nameasn::$fasn(&mut res, rhs);
                res
            }
        }
    };
    ($this:ident, $(($name:ident, $f:ident, $nameasn:ident, $fasn:ident)),*) => {
        $(
            impl<T: Primitive> $nameasn<T> for $this<T> {
                fn $fasn(&mut self, rhs: T) {
                    for a in self.iter_mut() {
                        *a = $name::$f(*a, rhs);
                    }
                }
            }

            impl<T: Primitive> $nameasn<&T> for $this<T> {
                fn $fasn(&mut self, rhs: &T) {
                    for a in self.iter_mut() {
                        *a = $name::$f(*a, *rhs);
                    }
                }
            }

            impl_primitive_op!(@variant $this<T>, ($name, $f, $nameasn, $fasn),  $this<T>,  T);
            impl_primitive_op!(@variant $this<T>, ($name, $f, $nameasn, $fasn),  $this<T>, &T);
            impl_primitive_op!(@variant $this<T>, ($name, $f, $nameasn, $fasn), &$this<T>,  T);
            impl_primitive_op!(@variant $this<T>, ($name, $f, $nameasn, $fasn), &$this<T>, &T);
        )*
    };
}

// Vector and vector operation implementation helper
macro_rules! impl_vector_op {
    (@variantasn $this:ty, ($name:ident, $f:ident, $nameasn:ident, $fasn:ident), $rhs:ty) => {
        impl<T: Primitive> $nameasn<$rhs> for $this {
            fn $fasn(&mut self, rhs: $rhs) {
                for (a, &b) in self.iter_mut().zip(rhs.iter()) {
                    *a = $name::$f(*a, b);
                }
            }
        }
    };
    (@variant $this:ty, ($name:ident, $f:ident, $nameasn:ident, $fasn:ident), $lhs:ty, $rhs:ty) => {
        impl<T: Primitive> $name<$rhs> for $lhs {
            type Output = $this;

            fn $f(self, rhs: $rhs) -> Self::Output {
                let mut res = self.clone();
                $nameasn::$fasn(&mut res, rhs);
                res
            }
        }
    };
    ($this:ident, $(($name:ident, $f:ident, $nameasn:ident, $fasn:ident)),*) => {
        $(
            impl_vector_op!(@variantasn $this<T>, ($name, $f, $nameasn, $fasn),             $this<T>);
            impl_vector_op!(@variantasn $this<T>, ($name, $f, $nameasn, $fasn),            &$this<T>);

            impl_vector_op!(@variant    $this<T>, ($name, $f, $nameasn, $fasn),  $this<T>,  $this<T>);
            impl_vector_op!(@variant    $this<T>, ($name, $f, $nameasn, $fasn),  $this<T>, &$this<T>);
            impl_vector_op!(@variant    $this<T>, ($name, $f, $nameasn, $fasn), &$this<T>,  $this<T>);
            impl_vector_op!(@variant    $this<T>, ($name, $f, $nameasn, $fasn), &$this<T>, &$this<T>);
        )*
    };
}

// Vector implentation helper
//
// We need this because we want column vectors and row vectors to be distinct, as in we can't add
// them directly, but they have the exact same methods and trait implementations. So we need to
// implement ColVector and RowVectors identically, and to reduce space we use this macro to do the
// implementation.
macro_rules! impl_vector {
    ($this:ident, $other:ident) => {
        impl<T: Primitive> Vector for $this<T> {
            type ElementType = T;

            /// New vector.
            fn new(elements: Vec<T>) -> Self {
                Self { elements }
            }

            fn elements(&self) -> &Vec<T> {
                &self.elements
            }

            fn elements_mut(&mut self) -> &mut Vec<T> {
                &mut self.elements
            }
        }

        impl<T: Primitive + Pow> Pow for $this<T> {
            fn pow<N: Into<f64> + Copy>(&self, n: N) -> Self {
                self.iter().map(move |a| a.pow(n)).collect()
            }
        }

        impl<T: Primitive + Random> $this<T> {
            pub fn random(n: usize) -> Self {
                Self::new(vec![T::random(); n])
            }
        }

        // Vector negation
        impl<T: Primitive> Neg for $this<T> {
            type Output = $this<T>;

            fn neg(self) -> Self::Output {
                self.iter().cloned().map(|a| -a).collect()
            }
        }

        // Vector reference negation
        impl<T: Primitive> Neg for &$this<T> {
            type Output = $this<T>;

            fn neg(self) -> Self::Output {
                -self.clone()
            }
        }

        // Vector and primitive operations (+, -, *, /)
        impl_primitive_op!(
            $this,
            (Add, add, AddAssign, add_assign),
            (Sub, sub, SubAssign, sub_assign),
            (Mul, mul, MulAssign, mul_assign),
            (Div, div, DivAssign, div_assign)
        );

        // Vector and vector operations (+, -)
        impl_vector_op!(
            $this,
            (Add, add, AddAssign, add_assign),
            (Sub, sub, SubAssign, sub_assign)
        );

        impl<'this, T: Primitive> FromIterator<T> for $this<T> {
            fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
                $this::new(iter.into_iter().collect())
            }
        }

        impl<T: Primitive> Conjugate for $this<T> {
            type Output = $other<T>;

            fn conj(&self) -> Self::Output {
                self.iter().map(|a| a.conj()).collect()
            }
        }

        impl<T: Primitive + Display> Display for $this<T> {
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

// Implement column vectors (ColVector) and row vectors (RowVector)
// The first argument is the type to be implemented, and the second is the return type of
// conjugation (column vector to row vector, row vector to column vector).
impl_vector!(ColVector, RowVector);
impl_vector!(RowVector, ColVector);

impl<A, B> Mul<&ColVector<B>> for &RowVector<A>
where
    A: Primitive + Mul<B>,
    B: Primitive,
    <A as Mul<B>>::Output: Primitive,
{
    type Output = <A as Mul<B>>::Output;

    fn mul(self, rhs: &ColVector<B>) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(&a, &b)| a * b).sum()
    }
}

impl<A, B> Mul<&ColVector<B>> for RowVector<A>
where
    A: Primitive + Mul<B>,
    B: Primitive,
    <A as Mul<B>>::Output: Primitive,
{
    type Output = <A as Mul<B>>::Output;

    fn mul(self, rhs: &ColVector<B>) -> Self::Output {
        Mul::mul(&self, rhs)
    }
}

impl<A, B> Mul<ColVector<B>> for &RowVector<A>
where
    A: Primitive + Mul<B>,
    B: Primitive,
    <A as Mul<B>>::Output: Primitive,
{
    type Output = <A as Mul<B>>::Output;

    fn mul(self, rhs: ColVector<B>) -> Self::Output {
        Mul::mul(self, &rhs)
    }
}

impl<A, B> Mul<ColVector<B>> for RowVector<A>
where
    A: Primitive + Mul<B>,
    B: Primitive,
    <A as Mul<B>>::Output: Primitive,
{
    type Output = <A as Mul<B>>::Output;

    fn mul(self, rhs: ColVector<B>) -> Self::Output {
        Mul::mul(&self, &rhs)
    }
}

// macro_rules! impl_op_scalar {
//     ($op:tt, $opassign:tt) => {
//         impl_op_ex!($opassign |lhs: &mut $this, rhs: &T| {
//             lhs.iter_mut().for_each(|a| *a = (*a) $op rhs);
//         });
//         impl_op_ex_commutative!($op |lhs: &$this, rhs: &T| -> $this {
//             let mut res = lhs.clone();
//             res $opassign rhs;
//             res
//         });
//     };
// }
//
// // Vector and scalar operations
// impl_op_scalar!(+, +=);
// impl_op_scalar!(-, -=);
// impl_op_scalar!(*, *=);
// impl_op_scalar!(/, /=);
//
// // Vector negation
// impl_op_ex!(-|v: &$this| -> $this {
//     v.clone() * (-1.0)
// });
//
// macro_rules! impl_op_vector {
//     ($op:tt, $opassign:tt) => {
//         impl_op_ex!($opassign |lhs: &mut $this, rhs: &$this| {
//             lhs.iter_mut().zip(rhs.iter()).for_each(|(a, b)| *a = (*a) $op (*b));
//         });
//         impl_op_ex!($op |lhs: &$this, rhs: &$this| -> $this {
//             let mut res = lhs.clone();
//             res $opassign rhs;
//             res
//         });
//     };
// }

// Vector and vector operations
//impl_op_vector!(+, +=);
//impl_op_vector!(-, -=);

//#[macro_export]
//macro_rules! cvecr {
//($($a:expr),*) => { ColVector::new(&vec![$($a as f64),*]) };
//}

// Row vector and matrix multiplication.
impl<A, B> Mul<&Matrix<B>> for &RowVector<A>
where
    A: Primitive + Mul<B>,
    B: Primitive,
    <A as Mul<B>>::Output: Primitive,
{
    type Output = RowVector<<A as Mul<B>>::Output>;

    fn mul(self, rhs: &Matrix<B>) -> Self::Output {
        use std::iter::repeat;

        assert_eq!(self.len(), rhs.shape().m());
        rhs.col_iter()
            .map(|ci| self.iter().zip(ci).map(|(&a, &b)| a * b).sum())
            .collect()
    }
}

// Matrix and column vector multiplication.
impl<A, B> Mul<&ColVector<B>> for &Matrix<A>
where
    A: Primitive + Mul<B>,
    B: Primitive,
    <A as Mul<B>>::Output: Primitive,
{
    type Output = ColVector<<A as Mul<B>>::Output>;

    fn mul(self, rhs: &ColVector<B>) -> Self::Output {
        assert_eq!(self.shape().n(), rhs.len());
        self.row_iter()
            .map(|ri| ri.zip(rhs.iter()).map(|(&a, &b)| a * b).sum())
            .collect()
    }
}
