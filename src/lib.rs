//#![allow(dead_code, unused_variables, unused_mut, unused_imports)]
#![feature(bool_to_option)]
#![feature(iterator_fold_self)]
#![allow(macro_expanded_macro_exports_accessed_by_absolute_paths)]

pub mod vector;
//use vector::Vector;

#[macro_use]
pub mod matrix;
use matrix::Matrix;

pub mod prelude {
    pub use crate::{
        matrix::{AsMatrixShape, Coord, Matrix, MatrixCoordinate, MatrixShape, Shape},
        vector::{ColVector, RowVector},
        {cvec, mat, rvec},
    };
}

pub type Space = Vec<Matrix>;

pub trait Orthogonalization {
    fn orthogonalize(vectors: &Space) -> Space;
}

pub struct GramSchmidt;

impl Orthogonalization for GramSchmidt {
    fn orthogonalize(vectors: &Space) -> Space {
        for _v in vectors {
            //let mut vj =
            //
        }
        vec![]
    }
}
