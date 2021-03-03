use crate::matrix::{Matrix, Shape};
use crate::norm::{Norm, TWO_NORM};
use crate::traits::Primitive;

/// QR decomposition using classical Gram-Schmidt orthogonalization.
pub fn qr_cgs<T: Primitive>(a: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    let &Shape(m, n) = a.shape();
    let mut q = Matrix::zero((m, n));
    let mut r = Matrix::zero(n);
    let mut v = a.clone();

    for j in 0..n {
        for i in 0..j {
            let rij = q.jth_col(i).conj() * a.jth_col(j);
            v.set_jth_col(j, v.jth_col(j) - q.jth_col(i) * rij);
            *(r.get_mut((i, j)).unwrap()) = rij;
        }
        let rjj = TWO_NORM.norm(&v.jth_col(j));
        *(r.get_mut((j, j)).unwrap()) = rjj;
        q.set_jth_col(j, v.jth_col(j) / rjj);
    }

    (q, r)
}

pub fn qr_mgs<T: Primitive>(a: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    let &Shape(m, n) = a.shape();
    //let mut q = Matrix::zero((m, n));
    //let mut r = q.clone();
    //let mut v = a.clone();

    //for i in 0..n {
    //let rii = TWO_NORM.norm(&a.jth_col(i));
    //*(r.get_mut((i, i)).unwrap()) = rii;
    //q.set_jth_col(i, v.jth_col(i) / rii);
    //for j in (i + 1)..n {
    //let rij = q.jth_col(j).conj() * v.jth_col(j);
    //*(r.get_mut((i, j)).unwrap()) = rij;
    //v.set_jth_col(j, v.jth_col(j) - q.jth_col(i) * rij);
    //}
    //}
    let mut q = Matrix::zero((m, n));
    let mut r = Matrix::zero(n);
    let mut v = a.clone();

    for j in 0..n {
        for i in 0..j {
            let rij = q.jth_col(i).conj() * v.jth_col(j);
            v.set_jth_col(j, v.jth_col(j) - q.jth_col(i) * rij);
            *(r.get_mut((i, j)).unwrap()) = rij;
        }
        let rjj = TWO_NORM.norm(&v.jth_col(j));
        *(r.get_mut((j, j)).unwrap()) = rjj;
        q.set_jth_col(j, v.jth_col(j) / rjj);
    }

    (q, r)
}
