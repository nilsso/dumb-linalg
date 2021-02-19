#![allow(unused_imports, unused_variables, unused_mut, dead_code)]

use hw03::prelude::*;
use itertools::iproduct;
use nalgebra::base::DMatrix;
use nalgebra::linalg::QR;

/// Classical Gram-Schmidt orthogonalization
///
/// * `input` - Slice of column vectors to orthogonalize.
fn orthogonalize(input: &[ColVector]) -> Vec<ColVector> {
    let m = input.get(0).unwrap().len();
    let n = input.len();

    let mut output: Vec<ColVector> = vec![];
    for i in 0..n {
        let v = {
            if i < n {
                let v = input.get(i).unwrap();
                if v.iter().all(|a| a - f64::EPSILON < 0.0) {
                    ColVector::random(m)
                } else {
                    v.clone() // bad, but works for now
                }
            } else {
                ColVector::random(m)
            }
        };

        let mut u = v.clone();
        for u_prev in output.iter().take(i) {
            u -= v.proj(u_prev);
        }
        output.push(u.normed());
    }
    output
}

/// QR-factorization via Gram-Schmidt orthogonalization
///
/// * `input` - Slice of column vectors to orthogonalize.
fn qr(input: &[ColVector]) -> (Matrix, Matrix) {
    let m = input.get(0).unwrap().len();
    let n = input.len();

    let e = orthogonalize(input);
    let q = Matrix::from_cols(e.iter().cloned());

    let mut r_elements: Vec<f64> = vec![];
    for j in 0..n {
        for i in 0..n {
            if i > j {
                r_elements.push(0.0);
            } else {
                let e = e.get(i).unwrap();
                let a = input.get(j).unwrap();
                r_elements.push(e.dot(&a));
            }
        }
    }
    let r = Matrix::from_iterator((n, n), r_elements.into_iter());

    (q, r)
}

fn test(rows: &[&[i32]]) {
    let m = rows.len();
    let n = rows[0].len();

    let a = Matrix::from_rows(
        rows.iter()
            .map(|r| RowVector::from_iterator(r.iter().map(|&a| a as f64))),
    );

    // Convert to column vectors
    let v: Vec<ColVector> = a
        .col_iter()
        .map(|c| ColVector::from_iterator(c.copied()))
        .collect();

    let (q, r) = qr(&v);
    println!("TEST, A = {}", a);
    println!("MY IMPLEMENTATION RESULTS:");
    println!("      A = {}", a);
    println!("      Q = {}", q);
    println!("      R = {}", r);
    println!("A-Q^T*R = {}", &a - &q * &r);
    println!("  Q*Q^T = {}", &q * q.t());

    let a = DMatrix::from_iterator(m, n, a.iter().cloned());
    let (q, r) = a.clone().qr().unpack();
    println!("NALGEBRA RESULTS:");
    println!("      A = {}", a);
    println!("      Q = {}", q);
    println!("      R = {}", r);
    println!("A-Q^T*R = {}", a - &q * &r);
    println!("  Q*Q^T = {}", &q * q.transpose());

    println!("----------------");
}

#[rustfmt::skip]
fn main() {
    test(&[
        &[1, 0],
        &[0, 1],
    ]);

    test(&[
        &[1, 2],
        &[3, 4],
    ]);

    test(&[
        &[8, -19],
        &[5,  30]
    ]);

    test(&[
        &[2, -4],
        &[5,  1],
        &[8,  6]
    ]);

    test(&[
        &[0, 0, 0, 1],
        &[0, 0, 1, 2],
        &[0, 1, 2, 3],
    ]);

    test(&[
        &[3],
        &[6],
        &[9],
    ]);
}
