#[cfg(test)]
mod complex_tests {
    use dumb_linalg::prelude::*;

    macro_rules! op_test {
        ($op:tt, $opassign:tt, $name1:ident, $name2:ident, $a:expr, $b:expr, $res:expr) => {
            #[test]
            fn $name1() {
                assert_eq!(&$a $op &$b, $res);
                assert_eq!($a $op $b, $res);
            }

            #[test]
            fn $name2() {
                let mut a = $a;
                a $opassign $b;
                assert_eq!(a, $res);
            }
        };
    }

    op_test!(+, +=,
        complex_add,
        complex_add_assign,
        cc!(3, 6), cc!(4, 8), cc!(7, 14));

    op_test!(-, -=,
        complex_sub,
        complex_sub_assign,
        cc!(3, 6), cc!(4, 8), cc!(-1, -2));

    op_test!(*, *=,
        complex_mul,
        complex_mul_assign,
        cc!(3, 6), cc!(4, 8), cc!(-36, 48));

    op_test!(/, /=,
        complex_div,
        complex_div_assign,
        cc!(3, 6), cc!(4, 8), cc!(3.0 / 4.0, 0.0));

    //#[test]
    //fn complex_trait_norm() {
    //assert_eq!(cc!(8, 15).norm(), 17.0); // 17 == sqrt(289) = sqrt(8^2 + 15^2)
    //}

    #[test]
    fn complex_trait_sqrt() {
        assert_eq!(cc!(15, 8).sqrt(), cc!(4, 1));
        assert_eq!(cc!(21, 20).sqrt(), cc!(5, 2));
        assert_eq!(cc!(24, 10).sqrt(), cc!(5, 1));
        assert_eq!(cc!(29, 23).sqrt(), cc!(29.0 / 2.0, 23.0 / 2.0));
    }

    #[test]
    fn complex_trait_conj() {
        assert_eq!(cc!(24, 10).conj(), cc!(24, -10));
    }
}
