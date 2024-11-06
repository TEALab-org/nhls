pub fn log2ceil(i: usize) -> usize {
    (usize::BITS - i.leading_zeros()) as usize
}


pub fn repeated_square(exp0: usize, i: usize) -> usize {
    let mut exp = exp0;
    let mut result = i;
    let mut v = 1;
    for _ in 0..log2ceil(exp0) {
        if exp % 2 == 1 {
            v *= result; 
            exp -= 1;
        }
        exp /= 2;
        result *= result;
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
}
