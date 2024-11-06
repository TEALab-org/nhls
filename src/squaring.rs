use num_traits::{Num, One};
use rayon::prelude::*;

pub fn log2ceil(i: usize) -> usize {
    (usize::BITS - i.leading_zeros()) as usize
}

// Reference implementation for integers
pub fn repeated_square(exp0: usize, i: usize) -> usize {
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

// Parallelized vector squaring
pub fn repeated_square_slice<NumType: Num + Copy + Send + std::marker::Sized>(
    exp0: usize,
    data: &mut [NumType],
    result_buffer: &mut [NumType],
    chunk_size: usize,
) {
    debug_assert!(data.len() == result_buffer.len());
    let mut exp = exp0;

    // Result buffer starts as 1
    result_buffer
        .par_chunks_mut(chunk_size)
        .for_each(|xs: &mut [NumType]| {
            for x in xs {
                *x = NumType::one();
            }
        });

    let n = log2ceil(exp0);
    for _ in 0..n {
        if exp % 2 == 1 {
            result_buffer
                .par_chunks_mut(chunk_size)
                .zip(data.par_chunks_mut(chunk_size))
                .for_each(|(rs, ds)| {
                    for (r, d) in rs.iter_mut().zip(ds.iter_mut()) {
                        *r = *r * *d;
                    }
                });

            //result *= data;
            exp -= 1;
        }
        exp /= 2;
        data.par_chunks_mut(chunk_size)
            .for_each(|xs: &mut [NumType]| {
                for x in xs {
                    *x = *x * *x;
                }
            });
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
        assert_eq!(repeated_square(2, 2), 4);
        assert_eq!(repeated_square(3, 2), 8);
        assert_eq!(repeated_square(4, 2), 16);
        assert_eq!(repeated_square(5, 2), 32);
        assert_eq!(repeated_square(9, 3), 19683);
    }

    #[test]
    fn repeated_square_slice_test() {
        {
            let mut data = vec![1, 2, 3, 4, 5];
            let mut buffer = vec![0; 5];
            repeated_square_slice(5, &mut data, &mut buffer, 1);
            for (i, x) in buffer.iter().enumerate() {
                assert_eq!(*x, (i + 1).pow(5));
            }
        }
        {
            let mut data = vec![1, 2, 3, 4, 5];
            let mut buffer = vec![0; 5];
            repeated_square_slice(5, &mut data, &mut buffer, 100);
            for (i, x) in buffer.iter().enumerate() {
                assert_eq!(*x, (i + 1).pow(5));
            }
        }
    }
}
