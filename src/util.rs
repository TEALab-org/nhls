pub use num_traits::{Num, One, Zero};

pub trait NumTrait = Num + Copy + Send + Sync;

pub type Bound<const GRID_DIMENSION: usize> = nalgebra::SVector<i32, { GRID_DIMENSION }>;
pub type Slopes<const GRID_DIMENSION: usize> = nalgebra::SMatrix<i32, { GRID_DIMENSION }, 2>;

pub fn real_buffer_size<const GRID_DIMENSION: usize>(space_size: &Bound<GRID_DIMENSION>) -> usize {
    let mut accumulator = 1;
    for d in space_size {
        accumulator *= *d as usize;
    }
    accumulator
}

pub fn complex_buffer_size<const GRID_DIMENSION: usize>(
    space_size: &Bound<GRID_DIMENSION>,
) -> usize {
    let mut accumulator = 1;
    let mut size_iter = space_size.iter().rev();
    accumulator *= *size_iter.next().unwrap() as usize / 2 + 1;
    for d in size_iter {
        accumulator *= *d as usize;
    }
    accumulator
}

pub fn linear_index<const GRID_DIMENSION: usize>(
    index: &Bound<GRID_DIMENSION>,
    bound: &Bound<GRID_DIMENSION>,
) -> usize {
    let mut accumulator = 0;
    for d in 0..GRID_DIMENSION {
        debug_assert!(index[d] >= 0);
        let mut dim_accumulator = index[d] as usize;
        for dn in (d + 1)..GRID_DIMENSION {
            dim_accumulator *= bound[dn] as usize;
        }
        accumulator += dim_accumulator;
    }
    accumulator
}

pub fn linear_to_coord<const GRID_DIMENSION: usize>(
    linear_index: usize,
    bound: &Bound<GRID_DIMENSION>,
) -> Bound<GRID_DIMENSION> {
    let mut result = Bound::zero();
    let mut index_accumulator = linear_index;

    for d in 0..GRID_DIMENSION - 1 {
        let mut dim_accumulator = 1;
        for dn in (d + 1)..GRID_DIMENSION {
            dim_accumulator *= bound[dn] as usize;
        }

        result[d] = (index_accumulator / dim_accumulator) as i32;
        index_accumulator %= dim_accumulator;
    }
    result[GRID_DIMENSION - 1] = index_accumulator as i32;
    result
}

pub fn periodic_offset_index<const GRID_DIMENSION: usize>(
    index: &Bound<GRID_DIMENSION>,
    bound: &Bound<GRID_DIMENSION>,
) -> Bound<GRID_DIMENSION> {
    let mut result = Bound::zero();
    for d in 0..GRID_DIMENSION {
        let di_raw = index[d];
        debug_assert!(di_raw < bound[d]);
        result[d] = if di_raw < 0 {
            bound[d] + di_raw
        } else {
            di_raw
        };
    }
    result
}

pub fn periodic_index<const GRID_DIMENSION: usize>(
    index: &Bound<GRID_DIMENSION>,
    bound: &Bound<GRID_DIMENSION>,
) -> Bound<GRID_DIMENSION> {
    let mut result = Bound::zero();
    for d in 0..GRID_DIMENSION {
        let di_raw = index[d];
        result[d] = if di_raw < 0 {
            bound[d] + di_raw
        } else if di_raw >= bound[d] {
            di_raw % bound[d]
        } else {
            di_raw
        }
    }
    result
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use nalgebra::vector;

    #[test]
    fn buffer_size_test() {
        {
            let dimensions = vector![5];
            let real_size = real_buffer_size(&dimensions);
            assert_eq!(real_size, 5);
            let complex_size = complex_buffer_size(&dimensions);
            assert_eq!(complex_size, (5 / 2) + 1);
        }

        {
            let dimensions = vector![5, 7, 9];
            let real_size = real_buffer_size(&dimensions);
            assert_eq!(real_size, 5 * 7 * 9);
            let complex_size = complex_buffer_size(&dimensions);
            assert_eq!(complex_size, 5 * 7 * ((9 / 2) + 1));
        }
    }

    #[test]
    fn linear_index_test() {
        {
            let index = vector![5, 7, 11];
            let bound = vector![20, 20, 20];
            assert_eq!(linear_index(&index, &bound), 5 * 20 * 20 + 7 * 20 + 11);
        }

        {
            let index = vector![5, 7];
            let bound = vector![20, 20];
            assert_eq!(linear_index(&index, &bound), 5 * 20 + 7);
        }

        {
            let index = vector![5];
            let bound = vector![20];
            assert_eq!(linear_index(&index, &bound), 5);
        }
    }

    #[test]
    fn periodic_offset_index_test() {
        {
            let index = vector![0, 0];
            let bound = vector![10, 10];
            assert_eq!(periodic_offset_index(&index, &bound), vector![0, 0]);
        }

        {
            let index = vector![-1, 0];
            let bound = vector![10, 10];
            assert_eq!(periodic_offset_index(&index, &bound), vector![9, 0]);
        }

        {
            let index = vector![0, -1];
            let bound = vector![10, 10];
            assert_eq!(periodic_offset_index(&index, &bound), vector![0, 9]);
        }

        {
            let index = vector![0, -1];
            let bound = vector![10, 10];
            assert_eq!(periodic_offset_index(&index, &bound), vector![0, 9]);
        }

        {
            let index = vector![0, -1, -4, -19, 34];
            let bound = vector![100, 100, 100, 100, 100];
            assert_eq!(
                periodic_offset_index(&index, &bound),
                vector![0, 99, 96, 81, 34]
            );
        }
    }

    #[test]
    fn periodic_index_test() {
        {
            let index = vector![0, 0];
            let bound = vector![10, 10];
            assert_eq!(periodic_index(&index, &bound), vector![0, 0]);
        }

        {
            let index = vector![-1, 0];
            let bound = vector![10, 10];
            assert_eq!(periodic_index(&index, &bound), vector![9, 0]);
        }

        {
            let index = vector![0, -1];
            let bound = vector![10, 10];
            assert_eq!(periodic_index(&index, &bound), vector![0, 9]);
        }

        {
            let index = vector![0, -1];
            let bound = vector![10, 10];
            assert_eq!(periodic_index(&index, &bound), vector![0, 9]);
        }

        {
            let index = vector![0, -1, -4, -19, 134];
            let bound = vector![100, 100, 100, 100, 100];
            assert_eq!(periodic_index(&index, &bound), vector![0, 99, 96, 81, 34]);
        }
    }

    #[test]
    fn linear_to_coord_test() {
        {
            let index = 67;
            let bound = vector![10, 10];
            assert_eq!(linear_to_coord(index, &bound), vector![6, 7]);
        }

        {
            let index = 67;
            let bound = vector![100];
            assert_eq!(linear_to_coord(index, &bound), vector![67]);
        }

        {
            let index = 0;
            let bound = vector![10, 10, 8, 10];
            assert_eq!(linear_to_coord(index, &bound), vector![0, 0, 0, 0]);
        }
    }
}
