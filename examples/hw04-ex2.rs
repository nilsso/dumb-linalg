use dumb_linalg::prelude::*;
use dumb_linalg::util::random_unitary;

fn print_data_row<T: Primitive + std::fmt::Display>(data: &Vec<T>) {
    for a in data.iter() {
        print!("{} ", a);
    }
    println!();
}

fn main() {
    let n = 80_usize;

    let mut singular_values = vec![0.5; n];
    for i in 1..n {
        singular_values[i] = singular_values[i - 1] / 2.0;
    }
    print_data_row(&singular_values);

    // Random unitary matrix with singular values 1/2, 1/2^2, ...
    let a = random_unitary(n, &singular_values);

    // Classical GS
    let (_q, r) = qr_cgs(&a);
    print_data_row(&r.diag());

    // Modified GS
    let (_q, r) = qr_mgs(&a);
    print_data_row(&r.diag());
}
