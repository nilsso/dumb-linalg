//#![allow(dead_code, unused_variables, unused_mut, unused_imports)]
#![allow(unused_attributes)]
#![feature(bool_to_option)]
#![feature(const_fn)]
#![feature(trait_alias)]

pub mod complex;
pub mod matrix;
pub mod norm;
pub mod qr;
pub mod traits;
pub mod util;
pub mod vector;

use complex::Complex;
use vector::{ColVector, RowVector};

pub type RColVector = ColVector<f64>;
pub type CColVector = ColVector<Complex>;
pub type RRowVector = RowVector<f64>;
pub type CRowVector = RowVector<Complex>;

pub mod prelude {
    #[rustfmt::skip]
    pub use crate::{
        complex::Complex,
        matrix::{Shape, Coord, Matrix, RMatrix, CMatrix},
        traits::{Conjugate, One, Primitive, Zero, Root, Pow},
        norm::{Norm, PNorm, TWO_NORM},
        vector::{ColVector, RowVector, Vector, InnerProduct, Project},
        qr::{qr_cgs, qr_mgs},
        {cc},
        {RColVector, CColVector, RRowVector, CRowVector},
        {rcvec, ccvec, rrvec, crvec},
        {rmat, cmat},
    };
}
