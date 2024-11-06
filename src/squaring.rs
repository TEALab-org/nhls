pub fn log2ceil(i: usize) -> usize {
    (usize::BITS - i.leading_zeros()) as usize
}

pub fn repeated_square(exp0: usize, i: usize) -> usize {
    let mut exp = exp0;
    let mut squares = i;
    let mut result = 1;
    let n = log2ceil(exp0);
    println!(
        "repeated, n: {}, exp: {}, result: {}, squares: {}",
        n, exp, result, squares
    );
    for _ in 0..n {
        if exp % 2 == 1 {
            result *= squares;
            exp -= 1;
            println!(
                "  - odd exp, result: {}, squares: {}, exp: {}",
                result, squares, exp
            );
        }
        exp /= 2;
        squares *= squares;
        println!(" * exp: {}, result: {}, squares: {}", exp, result, squares);
    }

    result
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn log2ceil_test() {
        assert_eq!(log2ceil(1), 1);
        assert_eq!(log2ceil(2), 2);
        assert_eq!(log2ceil(3), 2);
        assert_eq!(log2ceil(4), 3);
        assert_eq!(log2ceil(5), 3);
        assert_eq!(log2ceil(6), 3);
        assert_eq!(log2ceil(7), 3);
        assert_eq!(log2ceil(8), 4);
    }

    #[test]
    fn repeated_square_test() {
        assert_eq!(repeated_square(2, 2), 4);
        assert_eq!(repeated_square(3, 2), 8);
        assert_eq!(repeated_square(4, 2), 16);
        assert_eq!(repeated_square(5, 2), 32);
        assert_eq!(repeated_square(9, 3), 19683);
    }
}
