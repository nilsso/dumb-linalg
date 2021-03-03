use crate::matrix::{Matrix, Shape};
use crate::qr::qr_mgs;
use crate::traits::{Primitive, Random};

/// Product over iterator of primitives.
pub fn product<'a, T, I>(iter: I) -> T
where
    T: Primitive + 'a,
    I: Iterator<Item = &'a T>,
{
    iter.fold(T::ONE, |a, &b| a * b)
}

/// Generate a random unitary matrix.
pub fn random_unitary<T: Primitive + Random, S: Into<Shape>>(
    shape: S,
    singular_values: &Vec<T>,
) -> Matrix<T> {
    let shape = shape.into();
    assert_eq!(shape.m(), singular_values.len());
    let (u, _) = qr_mgs(&Matrix::<T>::random(shape));
    let (v, _) = qr_mgs(&Matrix::<T>::random(shape));
    let s = Matrix::<T>::from_diag(shape, &singular_values);

    u * s * v
}

/// Equidistantly
pub fn linspace(range: std::ops::Range<usize>, n: usize) -> impl Iterator<Item = f64> {
    let d = (range.end as f64 - range.start as f64) / (n - 1) as f64;
    let s = range.start as f64;
    (0..n).map(move |i| s + i as f64 * d)
}
