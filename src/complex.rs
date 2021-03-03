#![rustfmt::skip::macros(impl_op_ex)]
use std::cmp::PartialEq;
use std::convert::{From, Into};
use std::iter::Sum;
use std::{fmt, fmt::Display};

use auto_ops::{impl_op_ex, impl_op_ex_commutative};

use crate::traits::{Conjugate, One, Pow, Random, Root, Zero};

/// Complex number.
#[derive(Copy, Clone, Debug)]
pub struct Complex(pub f64, pub f64);

/// Complex constructor helper
#[macro_export]
macro_rules! cc {
    ($real_part:expr, $imaginary_part:expr) => {
        Complex($real_part as f64, $imaginary_part as f64)
    };
    ($real_part:expr) => {
        cc!($real_part, 0.0)
    };
}

macro_rules! impl_into {
    ($($u:ty),*) => {
        $(
            impl From<$u> for Complex {
                fn from(n: $u) -> Complex {
                    Complex(n as f64, 0.0)
                }
            }
        )*
    };
}

// Primitive to complex
impl_into!(u8, u16, u32, i8, i16, i32, f32, f64);

// Tuple to complex
impl<A: Into<f64>, B: Into<f64>> From<(A, B)> for Complex {
    fn from((r, i): (A, B)) -> Complex {
        Complex(r.into(), i.into())
    }
}

// Complex negation
impl_op_ex!(- |this: &Complex| -> Complex {
    Complex(-this.0, -this.1)
});

impl_op_ex!(+ |lhs: &Complex, rhs: &Complex| -> Complex {
    Complex(lhs.0 + rhs.0, lhs.1 + rhs.1)
});

impl_op_ex!(- |lhs: &Complex, rhs: &Complex| -> Complex {
    lhs + (-rhs)
});

impl_op_ex!(*|lhs: &Complex, rhs: &Complex| -> Complex {
    let Complex(a, b) = lhs;
    let Complex(c, d) = rhs;
    Complex(a * c - b * d, a * d + b * c)
});

impl_op_ex!(/ |lhs: &Complex, rhs: &Complex| -> Complex {
    let Complex(a, b) = lhs;
    let Complex(c, d) = rhs;
    let q = c * c + d * d;
    Complex((a * c + b * d) / q, (b * c - a * d) / q)
});

macro_rules! impl_op_assign {
    ($(($op:tt, $opassign:tt)),*) => {
        $(
            impl_op_ex!($opassign |lhs: &mut Complex, rhs: &Complex| {
                (*lhs) = (*lhs) $op (*rhs);
            });
        )*
    };
}

// Complex and complex operator assignment
impl_op_assign!((+, +=), (-, -=), (*, *=), (/, /=));

macro_rules! impl_op_primitive {
    (@op $u:path, $(( $op:tt, $opassign:tt)),*) => {
        $(
            impl_op_ex!($opassign |lhs: &mut Complex, rhs: &$u| {
                (*lhs) $opassign Into::<Complex>::into(*rhs);
            });

            impl_op_ex_commutative!($op |lhs: &Complex, rhs: &$u| -> Complex {
                lhs $op Into::<Complex>::into(*rhs)
            });
        )*
    };
    ($($u:path),*) => {
        $( impl_op_primitive!(@op $u, (+, +=), (-, -=), (*, *=), (/, /=)); )*
    };
}

// Complex and primitive operations (as operator assignment)
impl_op_primitive!(u8, u16, u32, i8, i16, i32, f32, f64);

impl Zero for Complex {
    const ZERO: Complex = Complex(0.0, 0.0);
}

impl One for Complex {
    const ONE: Complex = Complex(1.0, 0.0);
}

impl Pow for Complex {
    fn pow<N: Into<f64>>(&self, n: N) -> Complex {
        // DeMoivre's Theorem
        // c = a+bi = r*(cos(theta)+sin(theta)i)
        // r = sqrt(a^2 + b^2)
        // theta = atan2(b, a)
        // c^n = (a+bi)^2 = r^n*(cos(n*theta)+sin(n*theta)i)
        let n = n.into();
        let &Complex(a, b) = self;
        let r = (a * a + b * b).powf(n / 2.0);
        let theta = n * b.atan2(a);
        Complex(r * theta.cos(), r * theta.sin())
    }
}

impl Root for Complex {
    fn root(&self, n: i32) -> Complex {
        self.pow(1.0 / n as f64)
    }
}

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        (self.0 - other.0 < f64::EPSILON) && (self.1 - other.1 < f64::EPSILON)
    }
}

impl Conjugate for Complex {
    type Output = Complex;

    fn conj(&self) -> Complex {
        Complex(self.0, -self.1)
    }
}

impl Random for Complex {
    fn random() -> Complex {
        Complex(rand::random(), rand::random())
    }
}

impl Sum for Complex {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Complex::ZERO, |a, b| a + b)
    }
}

impl Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({})+({})i", self.0, self.1)?;
        Ok(())
    }
}
