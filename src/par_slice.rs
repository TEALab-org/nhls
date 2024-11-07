use num_traits::Num;
use rayon::prelude::*;

/// Sets each element to the same value.
/// `chunk_size` is break the work into tasks for multi-threading.
pub fn set_value<NumType: Num + Copy + Send + Sync>(
    a_slice: &mut [NumType],
    value: NumType,
    chunk_size: usize,
) {
    a_slice
        .par_chunks_mut(chunk_size)
        .for_each(|a_chunk: &mut [NumType]| {
            for a in a_chunk {
                *a = value;
            }
        });
}

pub fn square<NumType: Num + Copy + Send + Sync>(a_slice: &mut [NumType], chunk_size: usize) {
    a_slice
        .par_chunks_mut(chunk_size)
        .for_each(|a_chunk: &mut [NumType]| {
            for a in a_chunk {
                *a = *a * *a;
            }
        });
}

pub fn multiply_by<NumType: Num + Copy + Send>(
    a_slice: &mut [NumType],
    b_slice: &mut [NumType],
    chunk_size: usize,
) {
    a_slice
        .par_chunks_mut(chunk_size)
        .zip(b_slice.par_chunks_mut(chunk_size))
        .for_each(|(a_chunk, b_chunk)| {
            for (a, b) in a_chunk.iter_mut().zip(b_chunk.iter_mut()) {
                *a = *a * *b;
            }
        });
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn set_values_test() {
        {
            let mut a = vec![0, 1, 2, 3, 4, 5];
            set_value(&mut a, 7, 6);
            for v in a {
                assert_eq!(v, 7);
            }
        }

        {
            let n = 1000;
            let chunk_size = 10;
            let value = n + 1;
            let mut a = Vec::with_capacity(n);
            for i in 0..n {
                a.push(i);
            }

            set_value(&mut a, value, chunk_size);

            for v in a {
                assert_eq!(v, value);
            }
        }
    }
}
