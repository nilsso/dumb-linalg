/// Straightforward $m\times n$ matrix.
use auto_ops::*;
use itertools::iproduct;
use std::cmp::{PartialEq, PartialOrd};
use std::{fmt, fmt::Display};

use crate::vector::{ColVector, RowVector};

/// Matrix shape `(usize, usize)` alias.
pub type Shape = (usize, usize);
/// Matrix coordinate `(usize, usize)` alias.
pub type Coord = (usize, usize);

/// Matrix shape trait.
pub trait MatrixShape {
    fn m(&self) -> usize;
    fn n(&self) -> usize;
    fn len(&self) -> usize;
    fn t(&self) -> Self;
}

impl MatrixShape for Shape {
    fn m(&self) -> usize {
        self.0
    }

    fn n(&self) -> usize {
        self.1
    }

    fn len(&self) -> usize {
        let (m, n) = self;
        m * n
    }

    fn t(&self) -> Self {
        let (m, n) = self;
        (*n, *m)
    }
}

/// Type as matrix shape trait (`usize` or `(usize, usize)` to `Shape`).
pub trait AsMatrixShape {
    fn as_shape(self) -> Shape;
}

impl AsMatrixShape for usize {
    fn as_shape(self) -> Shape {
        (self, self)
    }
}

impl AsMatrixShape for (usize, usize) {
    fn as_shape(self) -> Shape {
        self
    }
}

/// Matrix flat index from coordinate.
pub fn index<S: AsMatrixShape>(coordinate: Coord, shape: S) -> usize {
    let (i, j) = coordinate;
    let (m, _) = shape.as_shape();
    j + i * m
}

/// Matrix coordinate trait.
pub trait MatrixCoordinate {
    fn to_index<S: AsMatrixShape>(self, shape: S) -> usize;
}

impl MatrixCoordinate for Coord {
    fn to_index<S: AsMatrixShape>(self, shape: S) -> usize {
        index(self, shape.as_shape())
    }
}

/// Straightforward $m\times n$ matrix.
///
/// Matrix with specified shape and `f64` elements.
#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct Matrix {
    shape: (usize, usize),
    elements: Vec<f64>,
}

/// Matrix constructor helper macro.
#[rustfmt::skip]
#[macro_export]
macro_rules! mat {
    ($shape:expr; $($a:expr),*)   => { Matrix::new($shape, &[$($a as f64),*]) };
}

impl Matrix {
    /// New matrix.
    ///
    /// * `shape` - Specified shape `(m, n)` ($m$ rows by $n$ columns).
    /// * `elements` - Specified matrix elements.
    ///
    /// Panics if the number of elements is not $mn$.
    pub fn new<S: AsMatrixShape>(shape: S, elements: &[f64]) -> Self {
        let shape = shape.as_shape();
        let mn = shape.len();
        if mn != elements.len() {
            panic!(
                "Shape {:?} does not match number of elements {}.",
                shape, mn
            );
        }
        Self {
            shape,
            elements: elements.into(),
        }
    }

    /// New matrix with $m$ rows.
    ///
    /// * `m` - Number of rows.
    /// * `elements` - Specified matrix.
    ///
    /// Panics if the number of elements is not $mn$.
    pub fn new_with_m(m: usize, elements: &[f64]) -> Self {
        Self::new((m, elements.len() / m), elements)
    }

    pub fn from_iterator<S, I>(shape: S, iter: I) -> Self
    where
        S: AsMatrixShape,
        I: Iterator<Item = f64>,
    {
        let shape = shape.as_shape();
        let elements: Vec<f64> = iter.collect();
        if shape.len() != elements.len() {
            panic!("Not enough elements for given matrix dimensions");
        }

        Self { shape, elements }
    }

    pub fn from_cols<'a, I>(mut iter: I) -> Self
    where
        I: Iterator<Item = ColVector>,
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
            shape: (m, n),
            elements,
        }
    }

    pub fn from_rows<'a, I>(iter: I) -> Self
    where
        I: Iterator<Item = RowVector>,
    {
        Matrix::from_cols(iter.map(|r| r.t())).t()
    }

    /// New zero matrix.
    ///
    /// * `shape` - Specified shape.
    pub fn zero<S: AsMatrixShape>(shape: S) -> Self {
        let shape = shape.as_shape();
        Self::new(shape, vec![0.0; shape.len()].as_slice())
    }

    /// New identity matrix.
    ///
    /// * `shape` - Specified shape.
    pub fn identity<S: AsMatrixShape>(shape: S) -> Self {
        let shape = shape.as_shape();
        let mut res = Self::zero(shape);
        let (m, n) = shape;
        for (i, j) in (0..m).zip(0..n) {
            *res.get_mut((i, j)).unwrap() = 1.0;
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
    pub fn shape(&self) -> &(usize, usize) {
        &self.shape
    }

    pub fn elements(&self) -> &[f64] {
        &self.elements
    }

    pub fn index(&self, coordinate: Coord) -> usize {
        coordinate.to_index(self.shape)
    }

    /// Iterator over the matrix elements.
    ///
    /// Returns an iterator over the underlying flat vector of matrix elements.
    pub fn iter(&self) -> std::slice::Iter<f64> {
        self.elements.iter()
    }

    /// Mutable iterator over the matrix elements.
    ///
    /// Returns a mutable iterator over the underlying flat vector of matrix elements.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<f64> {
        self.elements.iter_mut()
    }

    /// Iterator over the columns of the matrix.
    pub fn col_iter(&self) -> ColIter {
        ColIter::new(self)
    }

    /// Iterator over the rows of the matrix.
    pub fn row_iter(&self) -> RowIter {
        RowIter::new(self)
    }

    //pub fn row_iter(&self) -> std::slice::

    /// Get reference to the ${ij}^\text{th}$ element of the matrix.
    ///
    /// * `coordinate` - Matrix element coordinate.
    pub fn get(&self, coord: Coord) -> Option<&f64> {
        self.elements.get(coord.to_index(self.shape))
    }

    /// Get mutable reference to the ${ij}^\text{th}$ element of the matrix.
    ///
    /// * `coordinate` - Matrix element coordinate.
    pub fn get_mut(&mut self, coord: (usize, usize)) -> Option<&mut f64> {
        self.elements.get_mut(coord.to_index(self.shape))
    }

    /// Transpose this matrix.
    ///
    /// - A column matrix becomes a row matrix,
    /// - A row matrix, becomes a column matrix, and
    /// - A $m\times n$ matrix becomes $n\times m$ matrix.
    pub fn t(&self) -> Self {
        let (m, n) = self.shape;
        let elements = iproduct!(0..m, 0..n)
            .map(|(i, j)| self.get((j, i)).unwrap())
            .copied()
            .collect();
        Self {
            elements,
            shape: self.shape.t(),
        }
    }
}

// Matrix and scalar addition
#[rustfmt::skip]
impl_op_ex_commutative!(+ |lhs: &Matrix, rhs: &f64| -> Matrix {
    Matrix {
        elements: lhs.iter().copied().map(|a| a + rhs).collect(),
        .. *lhs
    }
});

// Matrix and scalar subtraction
#[rustfmt::skip]
impl_op_ex!(- |lhs: &Matrix, rhs: &f64| -> Matrix {
    lhs + (-rhs)
});

// Matrix and scalar multiplication
#[rustfmt::skip]
impl_op_ex_commutative!(* |lhs: &Matrix, rhs: &f64| -> Matrix {
    Matrix {
        elements: lhs.iter().copied().map(|a| a * rhs).collect(),
        ..*lhs
    }
});

// Matrix and scalar division
#[rustfmt::skip]
impl_op_ex!(/ |lhs: &Matrix, rhs: &f64| -> Matrix {
    lhs * (1.0 / rhs)
});

// Matrix negation
#[rustfmt::skip]
impl_op_ex!(-|v: &Matrix| -> Matrix {
    v.clone() * (-1.0)
});

// Matrix and matrix addition
#[rustfmt::skip]
impl_op_ex!(+ |lhs: &Matrix, rhs: &Matrix| -> Matrix {
    if lhs.shape != rhs.shape {
        panic!("Different matrix shapes")
    }
    Matrix {
        elements: lhs.iter().zip(rhs.iter()).map(|(a, b)| a + b).collect(),
        .. *lhs
    }
});

// Matrix and matrix subtraction
#[rustfmt::skip]
impl_op_ex!(- |lhs: &Matrix, rhs: &Matrix| -> Matrix {
    if lhs.shape != rhs.shape {
        panic!("Different matrix shapes")
    }
    lhs + (-rhs)
});

// Matrix and matrix multiplication
impl_op_ex!(*|lhs: &Matrix, rhs: &Matrix| -> Matrix {
    let l_shape = lhs.shape();
    let r_shape = rhs.shape();

    if l_shape.n() != r_shape.m() {
        panic!("Invalid matrix shapes")
    }

    let shape = (l_shape.m(), r_shape.n());
    let mut elements = vec![];

    for (_, r) in rhs.col_iter().enumerate() {
        for (_, l) in lhs.row_iter().enumerate() {
            elements.push(r.clone().zip(l).map(|(a, b)| a * b).sum())
        }
    }

    Matrix { shape, elements }
});

impl Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        #[rustfmt::skip]
        let (m, n) = self.shape;
        let elements = self.elements.as_slice();
        for i in 0..m {
            write!(f, "[")?;
            write!(f, "{:6.2}", elements[i])?;
            for j in 1..n {
                write!(f, " {:6.2}", elements[i + j * m])?;
            }
            write!(f, "]")?;
            if i < m - 1 {
                write!(f, ",")?;
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

/// Matrix column iterator.
pub struct ColIter<'a> {
    elements: &'a [f64],
    m: usize,
    n: usize,
    i: usize,
}

impl<'a> ColIter<'a> {
    pub fn new(matrix: &'a Matrix) -> Self {
        let &(m, n) = matrix.shape();

        Self {
            elements: matrix.elements(),
            m,
            n,
            i: 0,
        }
    }
}

impl<'a> Iterator for ColIter<'a> {
    type Item = std::iter::Take<std::iter::Skip<std::slice::Iter<'a, f64>>>;

    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            elements,
            m,
            n,
            ref mut i,
        } = self;
        if i < n {
            *i += 1;
            Some(elements.iter().skip((*i - 1) * *m).take(*m))
        } else {
            None
        }
    }
}

/// Matrix row iterator.
pub struct RowIter<'a> {
    elements: &'a [f64],
    m: usize,
    i: usize,
}

impl<'a> RowIter<'a> {
    pub fn new(matrix: &'a Matrix) -> Self {
        Self {
            elements: matrix.elements(),
            m: matrix.shape().m(),
            i: 0,
        }
    }
}

impl<'a> Iterator for RowIter<'a> {
    //type Item = &'a [f64];
    type Item = std::iter::StepBy<std::iter::Skip<std::slice::Iter<'a, f64>>>;

    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            elements,
            m,
            ref mut i,
        } = self;
        if i < m {
            *i += 1;
            Some(elements.iter().skip(*i - 1).step_by(*m))
        } else {
            None
        }
    }
}
