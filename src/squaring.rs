use crate::par_slice;
use num_traits::Num;
use rayon::prelude::*;

/// Basicly works out to `log2(i) + 1
pub fn log2ceil(i: usize) -> usize {
    (usize::BITS - i.leading_zeros()) as usize
}

/// Reference implementation of repeated squares for positive integers
pub fn repeated_square_reference(exp0: usize, i: usize) -> usize {
    let mut exp = exp0;
    let mut squares = i;
    let mut result = 1;
    let n = log2ceil(exp0);
    for _ in 0..n {
        if exp % 2 == 1 {
            result *= squares;
            exp -= 1;
        }
        exp /= 2;
        squares *= squares;
    }

    result
}

/// Parallelized squaring of slice elements.
/// Implements the following recursion description:
///
///
/// x^n = { x * (x^2)^((n -1)/2)
///       { x * (x2)^k
/// Our base case if $n
pub fn repeated_square<NumType: Num + Copy + Send + Sync + std::marker::Sized>(
    exp0: usize,
    data: &mut [NumType],
    result_buffer: &mut [NumType],
    chunk_size: usize,
) {
    debug_assert!(data.len() == result_buffer.len());
    let mut exp = exp0;

    // Result buffer starts as 1
    par_slice::set_value(result_buffer, NumType::one(), chunk_size);

    let n = log2ceil(exp0);
    for _ in 0..n {
        if exp % 2 == 1 {
            par_slice::multiply_by(result_buffer, data, chunk_size);

            //result *= data;
            exp -= 1;
        }
        exp /= 2;
        par_slice::square(data, chunk_size);
    }
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
        assert_eq!(repeated_square_reference(2, 2), 4);
        assert_eq!(repeated_square_reference(3, 2), 8);
        assert_eq!(repeated_square_reference(4, 2), 16);
        assert_eq!(repeated_square_reference(5, 2), 32);
        assert_eq!(repeated_square_reference(9, 3), 19683);
    }

    #[test]
    fn repeated_square_slice_test() {
        {
            let mut data = vec![1, 2, 3, 4, 5];
            let mut buffer = vec![0; 5];
            repeated_square(5, &mut data, &mut buffer, 1);
            for (i, x) in buffer.iter().enumerate() {
                assert_eq!(*x, (i + 1).pow(5));
            }
        }
        {
            let mut data = vec![1, 2, 3, 4, 5];
            let mut buffer = vec![0; 5];
            repeated_square(5, &mut data, &mut buffer, 100);
            for (i, x) in buffer.iter().enumerate() {
                assert_eq!(*x, (i + 1).pow(5));
            }
        }
    }
}
