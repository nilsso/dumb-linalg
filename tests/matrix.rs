#[cfg(test)]
mod matrix_tests {
    use dumb_linalg::prelude::*;

    #[test]
    fn col_iterator() {
        let a = rmat![3; 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let res = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        for (li, ri) in a.col_iter().zip(res.chunks(3)) {
            for (l, r) in li.zip(ri) {
                assert_eq!(l, r);
            }
        }
    }

    #[test]
    fn row_iterator() {
        let a = rmat![3; 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let res = &[1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

        for (li, ri) in a.row_iter().zip(res.chunks(3)) {
            for (l, r) in li.zip(ri) {
                assert_eq!(l, r);
            }
        }
    }
}
