use dumb_linalg::prelude::*;
use dumb_linalg::util::linspace;

fn main() {
    let x: ColVector<f64> = linspace(0..10, 128).collect();
    let _a = Matrix::from_cols(vec![x.pow(0), x.pow(1), x.pow(2), x.pow(3)].into_iter());
}
