use crate::traits::Primitive;
use crate::vector::Vector;

/// Generic norm.
pub trait Norm<T: Primitive> {
    fn norm(&self, v: &dyn Vector<ElementType = T>) -> T;
}

pub struct PNorm {
    p: i32,
}

impl PNorm {
    pub fn new(p: i32) -> Self {
        Self { p }
    }
}

impl<T: Primitive> Norm<T> for PNorm {
    fn norm(&self, v: &dyn Vector<ElementType = T>) -> T {
        v.iter().map(|&a| a.pow(self.p)).sum::<T>().root(self.p)
    }
}

pub const TWO_NORM: PNorm = PNorm { p: 2 };
