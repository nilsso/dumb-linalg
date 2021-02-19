use crate::matrix::*;

pub trait Norm {
    fn norm();
    fn normalized();
}

struct PNorm;

impl Norm for PNorm {}

/// Two-norm of a column or row vector.
pub fn magnitude(&self) -> Option<f64> {
    self.iter().sum::<f64>().sqrt()
}

pub fn normalized(&self) -> Self {
    let m: f64 = self.magnitude();
    Self {
        elements: self.iter().cloned().map(|a| a / m).collect(),
        ..*self
    }
}
