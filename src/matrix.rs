//#![allow(unused_imports)]
/// Straightforward $m\times n$ matrix.
//use auto_ops::*;
use itertools::iproduct;

use std::cmp::{PartialEq, PartialOrd};
use std::convert::Into;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::{fmt, fmt::Display};
//use std::ops::{Index, IndexMut, Range, RangeFrom, RangeFull, RangeTo};

use crate::complex::Complex;
use crate::traits::{Conjugate, Primitive, Random};
use crate::util::product;
use crate::vector::{ColVector, RowVector, Vector};
//use crate::

/// Matrix shape `(usize, usize)` alias.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Shape(pub usize, pub usize);

impl Shape {
    pub fn m(&self) -> usize {
        self.0
    }

    pub fn n(&self) -> usize {
        self.1
    }

    pub fn len(&self) -> usize {
        self.0 * self.1
    }

    pub fn is_square(&self) -> bool {
        self.0 == self.1
    }

    pub fn t(&self) -> Self {
        Shape(self.1, self.0)
    }
}

impl Into<Shape> for usize {
    fn into(self) -> Shape {
        Shape(self, self)
    }
}

impl Into<Shape> for (usize, usize) {
    fn into(self) -> Shape {
        Shape(self.0, self.1)
    }
}

/// Matrix flat index from coordinate.
pub fn index<C: Into<Coord>, S: Into<Shape>>(coordinate: C, shape: S) -> usize {
    let Coord(i, j) = coordinate.into();
    let Shape(m, _) = shape.into();
    i + j * m
}

/// Matrix coordinate `(usize, usize)` alias.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Coord(usize, usize);

impl Coord {
    pub fn to_index<S: Into<Shape>>(self, shape: S) -> usize {
        index(self, shape)
    }
}

impl Into<Coord> for usize {
    fn into(self) -> Coord {
        Coord(self, self)
    }
}

impl Into<Coord> for (usize, usize) {
    fn into(self) -> Coord {
        Coord(self.0, self.1)
    }
}

/// Straightforward $m\times n$ matrix.
///
/// Matrix with specified shape and `f64` elements.
#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct Matrix<T: Primitive> {
    shape: Shape,
    elements: Vec<T>,
}

pub type RMatrix = Matrix<f64>;
pub type CMatrix = Matrix<Complex>;

#[macro_export]
macro_rules! rmat {
    ($shape:expr; $($a:expr),+$(,)*)   => { RMatrix::new($shape, vec![$($a as f64),*]) };
}

#[macro_export]
macro_rules! cmat {
    ($shape:expr; $($a:expr),+$(,)*)   => { CMatrix::new($shape, vec![$($a.into()),*]) };
}

impl<T: Primitive> Matrix<T> {
    /// New matrix.
    ///
    /// * `shape` - Specified shape `(m, n)` ($m$ rows by $n$ columns).
    /// * `elements` - Specified matrix elements.
    ///
    /// Panics if the number of elements is not $mn$.
    pub fn new<S: Into<Shape>>(shape: S, elements: Vec<T>) -> Self {
        let shape = shape.into();
        let mn = shape.len();
        if mn != elements.len() {
            panic!(
                "Shape {:?} does not match number of elements {}.",
                shape, mn
            );
        }
        Self { shape, elements }
    }

    /// New matrix with $m$ rows.
    ///
    /// * `m` - Number of rows.
    /// * `elements` - Specified matrix.
    ///
    /// Panics if the number of elements is not $mn$.
    pub fn new_with_m(m: usize, elements: Vec<T>) -> Self {
        Self::new((m, elements.len() / m), elements)
    }

    pub fn from_iterator<S, I>(shape: S, iter: I) -> Self
    where
        S: Into<Shape>,
        I: Iterator<Item = T>,
    {
        let shape = shape.into();
        let elements: Vec<T> = iter.collect();
        if shape.len() != elements.len() {
            panic!("Not enough elements for given matrix dimensions");
        }

        Self { shape, elements }
    }

    pub fn from_diag<S: Into<Shape>>(shape: S, diag_elements: &Vec<T>) -> Self {
        let shape = shape.into();
        assert_eq!(shape.0.min(shape.1), diag_elements.len());
        let mut res = Self::zero(shape);
        for (a, &b) in res.diag_iter_mut().zip(diag_elements.iter()) {
            *a = b;
        }
        res
    }

    pub fn from_cols<'a, I>(mut iter: I) -> Self
    where
        I: Iterator<Item = ColVector<T>>,
    {
        let mut elements = vec![];
        let mut m = 0;
        let mut n = 0;
        if let Some(col) = iter.next() {
            m = col.len();
            n += 1;
            elements.extend(col.iter());
            while let Some(col) = iter.next() {
                if col.len() != m {
                    panic!("Rows must all have same size");
                }
                n += 1;
                elements.extend(col.iter());
            }
        }
        Self {
            shape: Shape(m, n),
            elements,
        }
    }

    pub fn from_rows<I>(iter: I) -> Self
    where
        I: Iterator<Item = RowVector<T>>,
    {
        Matrix::from_cols(iter.map(|r| r.conj())).conj()
    }

    /// New zero matrix.
    ///
    /// * `shape` - Specified shape.
    pub fn zero<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        Self::new(shape, vec![T::ZERO; shape.len()])
    }

    /// New identity matrix.
    ///
    /// * `shape` - Specified shape.
    pub fn identity<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        let Shape(m, n) = shape;
        let mut res = Self::zero(shape);
        for (i, j) in (0..m).zip(0..n) {
            *res.get_mut((i, j)).unwrap() = T::ONE;
        }
        res
    }

    /// Total number of elements.
    ///
    /// Returns the number of elements of the underlying flat vector of matrix elements.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Shape of this matrix.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn is_square(&self) -> bool {
        self.shape.is_square()
    }

    pub fn m(&self) -> usize {
        self.shape.m()
    }

    pub fn n(&self) -> usize {
        self.shape.n()
    }

    pub fn elements(&self) -> &[T] {
        &self.elements
    }

    pub fn index<C: Into<Coord>>(&self, coordinate: C) -> usize {
        coordinate.into().to_index(self.shape)
    }

    /// Iterator over the matrix elements.
    ///
    /// Returns an iterator over the underlying flat vector of matrix elements.
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.elements.iter()
    }

    /// Mutable iterator over the matrix elements.
    ///
    /// Returns a mutable iterator over the underlying flat vector of matrix elements.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.elements.iter_mut()
    }

    /// Iterator over the columns of the matrix.
    pub fn col_iter(&self) -> ColIter<T> {
        ColIter::new(self)
    }

    /// Iterator over the rows of the matrix.
    pub fn row_iter(&self) -> RowIter<T> {
        RowIter::new(self)
    }

    /// Iterator over the diagonal entries.
    pub fn diag_iter(&self) -> impl Iterator<Item = &T> {
        let Shape(m, _) = self.shape;
        self.iter().step_by(m + 1)
    }

    /// Mutable iterator over the diagonal entries.
    pub fn diag_iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T> {
        let Shape(m, _) = self.shape;
        self.iter_mut().step_by(m + 1)
    }

    pub fn diag(&self) -> Vec<T> {
        self.diag_iter().copied().collect()
    }

    /// Get reference to the ${ij}^\text{th}$ element of the matrix.
    ///
    /// * `coordinate` - Matrix element coordinate.
    pub fn get<C: Into<Coord>>(&self, coord: C) -> Option<&T> {
        self.elements.get(coord.into().to_index(self.shape))
    }

    /// Get mutable reference to the ${ij}^\text{th}$ element of the matrix.
    ///
    /// * `coordinate` - Matrix element coordinate.
    pub fn get_mut<C: Into<Coord>>(&mut self, coord: C) -> Option<&mut T> {
        self.elements.get_mut(coord.into().to_index(self.shape))
    }

    fn ith_row_iter(&self, i: usize) -> impl Iterator<Item = &T> {
        let m = self.m();
        self.iter().skip(i).step_by(m)
    }

    fn ith_row_iter_mut(&mut self, i: usize) -> impl Iterator<Item = &mut T> {
        let m = self.m();
        self.iter_mut().skip(i).step_by(m)
    }

    /// Get the $i^\mathrm{th}$ row of this matrix.
    pub fn ith_row(&self, i: usize) -> RowVector<T> {
        self.ith_row_iter(i).copied().collect()
    }

    /// Set the $i^\mathrm{th}$ row of this matrix.
    pub fn set_ith_row(&mut self, i: usize, row: RowVector<T>) {
        for (a, &b) in self.ith_row_iter_mut(i).zip(row.iter()) {
            *a = b;
        }
    }

    fn jth_col_iter(&self, j: usize) -> impl Iterator<Item = &T> {
        let m = self.m();
        self.iter().skip(j * m).take(m)
    }

    fn jth_col_iter_mut(&mut self, j: usize) -> impl Iterator<Item = &mut T> {
        let m = self.m();
        self.iter_mut().skip(j * m).take(m)
    }

    /// Get the $j^\mathrm{th}$ column of this matrix.
    pub fn jth_col(&self, j: usize) -> ColVector<T> {
        self.jth_col_iter(j).copied().collect()
    }

    /// Set the $j^\mathrm{th}$ column of this matrix.
    pub fn set_jth_col(&mut self, j: usize, col: ColVector<T>) {
        for (a, &b) in self.jth_col_iter_mut(j).zip(col.iter()) {
            *a = b;
        }
    }

    /// Return a submatrix copy of this matrix.
    pub fn submatrix<Ctl: Into<Coord>, Cbr: Into<Coord>>(&self, tl: Ctl, br: Cbr) -> Self {
        let tl = tl.into();
        let br = br.into();
        let shape = Shape(br.0 - tl.0 + 1, br.1 - tl.1 + 1);
        let elements = (tl.0..=br.0)
            .flat_map(|j| (tl.1..=br.1).map(move |i| self.get((i, j)).unwrap()))
            .copied()
            .collect();
        Matrix::new(shape, elements)
    }

    /// Return a copy of this matrix with the $j^\mathrm{th}$ column removed.
    pub fn remove_jth_col(&self, j: usize) -> Self {
        let Shape(m, n) = self.shape;
        let shape = Shape(m, n - 1);
        let elements = (0..j)
            .chain((j + 1)..n)
            .flat_map(|j| (0..m).map(move |i| self.get((i, j)).unwrap()))
            .copied()
            .collect();
        Matrix::new(shape, elements)
    }

    /// Return a copy of this matrix with the first row and $j^\mathrm{th}$ column removed.
    pub fn minor(&self, j: usize) -> Self {
        let Shape(m, n) = self.shape;
        let shape = Shape(m - 1, n - 1);
        let elements = (0..j)
            .chain((j + 1)..n)
            .flat_map(|j| (1..m).map(move |i| self.get((i, j)).unwrap()))
            .copied()
            .collect();
        Matrix::new(shape, elements)
    }

    /// Get the determinant of this matrix.
    pub fn determinant(&self) -> Option<T> {
        if !self.is_square() {
            return None;
        }
        let d = match self.shape {
            Shape(1, 1) => *self.iter().next().unwrap(),
            Shape(2, 2) => {
                let mut i = self.iter();
                let &a = i.next().unwrap();
                let &c = i.next().unwrap();
                let &b = i.next().unwrap();
                let &d = i.next().unwrap();
                a * d - b * c
            }
            Shape(m, _) => {
                //(0..m)
                //.map(|j| {
                //let sign = if j % 2 == 0 { T::ONE } else { -T::ONE };
                //sign * self.remove_jth_col(j).determinant()
                //})
                //.sum()
                [T::ONE, -T::ONE]
                    .iter()
                    .cycle()
                    .zip(0..m)
                    .map(|(&sign, j)| {
                        let &a = self.get((0, j)).unwrap();
                        a * self.minor(j).determinant().unwrap() * sign
                    })
                    .sum()
            }
        };
        Some(d)
    }

    pub fn is_triangular(&self) -> bool {
        self.is_square() && product(self.diag_iter()) == self.determinant().unwrap()
    }
}

impl<T: Primitive + Random> Matrix<T> {
    pub fn random<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        Self::new(shape, (0..shape.len()).map(|_| T::random()).collect())
    }
}

impl<T: Primitive> Conjugate for Matrix<T> {
    type Output = Self;

    fn conj(&self) -> Self::Output {
        let Shape(m, n) = self.shape;
        let elements = iproduct!(0..m, 0..n)
            .map(|(j, i)| self.get((j, i)).unwrap())
            .copied()
            .collect();
        Self {
            elements,
            shape: self.shape.t(),
        }
    }
}

macro_rules! impl_primitive_op {
    ($(($name:ident, $f:ident, $nameasn:ident, $fasn:ident)),*) => {
        $(
            impl<T: Primitive> $nameasn<T> for Matrix<T> {
                fn $fasn(&mut self, scalar: T) {
                    for a in self.iter_mut() {
                        $nameasn::$fasn(a, scalar);
                    }
                }
            }
            impl<S, T> $name<S> for &Matrix<T>
            where
                S: Primitive,
                T: Primitive + $name<S>,
                <T as $name<S>>::Output: Primitive,
            {
                type Output = Matrix<<T as $name<S>>::Output>;

                fn $f(self, scalar: S) -> Self::Output {
                    let elements = self.iter().map(|&a| $name::$f(a, scalar)).collect();
                    Matrix::new(self.shape, elements)
                }
            }
            impl<S, T> $name<S> for Matrix<T>
            where
                S: Primitive,
                T: Primitive + $name<S>,
                <T as $name<S>>::Output: Primitive,
            {
                type Output = Matrix<<T as $name<S>>::Output>;

                fn $f(self, scalar: S) -> Self::Output {
                    $name::$f(&self, scalar)
                }
            }
        )*
    }
}

// Matrix and scalar operations
impl_primitive_op!(
    (Add, add, AddAssign, add_assign),
    (Sub, sub, SubAssign, sub_assign),
    (Mul, mul, MulAssign, mul_assign),
    (Div, div, DivAssign, div_assign)
);

// Matrix negation
impl<T: Primitive> Neg for Matrix<T> {
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        self * (-T::ONE)
    }
}
impl<T: Primitive> Neg for &Matrix<T> {
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        self.clone() * (-T::ONE)
    }
}

macro_rules! impl_matrix_matrix_op {
    ($name:ident, $f:ident) => {
        impl<A, B> $name<&Matrix<B>> for &Matrix<A>
        where
            A: Primitive + $name<B>,
            B: Primitive,
            <A as $name<B>>::Output: Primitive,
        {
            type Output = Matrix<<A as $name<B>>::Output>;

            fn $f(self, rhs: &Matrix<B>) -> Self::Output {
                assert_eq!(self.shape, rhs.shape);
                let elements = self
                    .iter()
                    .zip(rhs.iter())
                    .map(|(a, b)| $name::$f(*a, *b))
                    .collect();
                Matrix::new(self.shape, elements)
            }
        }

        impl<A, B> $name<&Matrix<B>> for Matrix<A>
        where
            A: Primitive + $name<B>,
            B: Primitive,
            <A as $name<B>>::Output: Primitive,
        {
            type Output = Matrix<<A as $name<B>>::Output>;

            fn $f(self, rhs: &Matrix<B>) -> Self::Output {
                $name::$f(&self, rhs)
            }
        }

        impl<A, B> $name<Matrix<B>> for &Matrix<A>
        where
            A: Primitive + $name<B>,
            B: Primitive,
            <A as $name<B>>::Output: Primitive,
        {
            type Output = Matrix<<A as $name<B>>::Output>;

            fn $f(self, rhs: Matrix<B>) -> Self::Output {
                $name::$f(self, &rhs)
            }
        }

        impl<A, B> $name<Matrix<B>> for Matrix<A>
        where
            A: Primitive + $name<B>,
            B: Primitive,
            <A as $name<B>>::Output: Primitive,
        {
            type Output = Matrix<<A as $name<B>>::Output>;

            fn $f(self, rhs: Matrix<B>) -> Self::Output {
                $name::$f(&self, &rhs)
            }
        }
    };
}

// Matrix and matrix addition and subtraction
impl_matrix_matrix_op!(Add, add);
impl_matrix_matrix_op!(Sub, sub);

// Matrix and matrix multiplication
impl<A, B> Mul<&Matrix<B>> for &Matrix<A>
where
    A: Primitive + Mul<B>,
    B: Primitive,
    <A as Mul<B>>::Output: Primitive,
{
    type Output = Matrix<<A as Mul<B>>::Output>;

    fn mul(self, rhs: &Matrix<B>) -> Self::Output {
        // self = (m x p)
        // rhs  = (p x n)
        // self * rhs = (m x n)
        // (m: columns of self, n: rows of rhs)
        assert!(self.shape().n() == rhs.shape().m());
        let shape = (self.shape().m(), rhs.shape().n());
        let elements = rhs
            .col_iter()
            .flat_map(|ci| {
                self.row_iter()
                    .map(move |ri| ri.zip(ci.clone()).map(|(a, b)| (*a) * (*b)).sum())
            })
            .collect();
        Matrix::new(shape, elements)
    }
}

impl<T: Primitive> Mul<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        &self * rhs
    }
}

impl<T: Primitive> Mul<Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        self * &rhs
    }
}

impl<T: Primitive> Mul<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        &self * &rhs
    }
}

impl<T: Primitive + Display> Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[\n")?;
        let Shape(m, n) = self.shape;
        for i in 0..m {
            write!(f, "[")?;
            write!(f, "{:6.2}", self.get((i, 0)).unwrap())?;
            for j in 1..n {
                write!(f, " {:6.2}", self.get((i, j)).unwrap())?;
            }
            write!(f, "]")?;
            if i < m - 1 {
                write!(f, ",\n")?;
            }
        }
        write!(f, "\n]")?;
        Ok(())
    }
}

/// Matrix column iterator.
pub struct ColIter<'a, T: Primitive> {
    elements: &'a [T],
    shape: Shape,
    i: usize,
}

impl<'a, T: Primitive> ColIter<'a, T> {
    pub fn new(matrix: &'a Matrix<T>) -> Self {
        Self {
            elements: matrix.elements(),
            shape: *matrix.shape(),
            i: 0,
        }
    }
}

impl<'a, T: Primitive> Iterator for ColIter<'a, T> {
    //type Item = impl Iterator<Item = T>;
    type Item = std::iter::Take<std::iter::Skip<std::slice::Iter<'a, T>>>;

    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            ref elements,
            ref shape,
            ref mut i,
        } = self;
        let &Shape(m, n) = shape;
        if (*i) < n {
            let iter = elements.iter().skip((*i) * m).take(m);
            *i += 1;
            Some(iter)
        } else {
            None
        }
    }
}

/// Matrix row iterator.
pub struct RowIter<'a, T: Primitive> {
    elements: &'a [T],
    shape: Shape,
    i: usize,
}

impl<'a, T: Primitive> RowIter<'a, T> {
    pub fn new(matrix: &'a Matrix<T>) -> Self {
        Self {
            elements: matrix.elements(),
            shape: *matrix.shape(),
            i: 0,
        }
    }
}

impl<'a, T: Primitive> Iterator for RowIter<'a, T> {
    type Item = std::iter::Take<std::iter::StepBy<std::iter::Skip<std::slice::Iter<'a, T>>>>;

    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            ref elements,
            ref shape,
            ref mut i,
        } = self;
        let &Shape(m, n) = shape;
        if (*i) < m {
            let iter = elements.iter().skip(*i).step_by(m).take(n);
            *i += 1;
            Some(iter)
        } else {
            None
        }
    }
}
