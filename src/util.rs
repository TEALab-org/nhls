pub use num_traits::{Num, One, Zero};

pub trait NumTrait = Num + Copy + Send + Sync;

pub type Coord<const GRID_DIMENSION: usize> = nalgebra::SVector<i32, { GRID_DIMENSION }>;
pub type Box<const GRID_DIMENSION: usize> = nalgebra::SMatrix<i32, { GRID_DIMENSION }, 2>;

pub fn box_buffer_size<const GRID_DIMENSION: usize>(view_box: &Box<GRID_DIMENSION>) -> usize {
    let diff = (view_box.column(1) - view_box.column(0)).add_scalar(1);
    real_buffer_size(&diff)
}

pub fn real_buffer_size<const GRID_DIMENSION: usize>(space_size: &Coord<GRID_DIMENSION>) -> usize {
    let mut accumulator = 1;
    for d in space_size {
        accumulator *= *d as usize;
    }
    accumulator
}

pub fn complex_buffer_size<const GRID_DIMENSION: usize>(
    space_size: &Coord<GRID_DIMENSION>,
) -> usize {
    let mut accumulator = 1;
    let mut size_iter = space_size.iter().rev();
    accumulator *= *size_iter.next().unwrap() as usize / 2 + 1;
    for d in size_iter {
        accumulator *= *d as usize;
    }
    accumulator
}

pub fn box_complex_buffer_size<const GRID_DIMENSION: usize>(
    view_box: &Box<GRID_DIMENSION>,
) -> usize {
    let diff = (view_box.column(1) - view_box.column(0)).add_scalar(1);
    complex_buffer_size(&diff)
}

pub fn linear_index<const GRID_DIMENSION: usize>(
    coord: &Coord<GRID_DIMENSION>,
    bound: &Coord<GRID_DIMENSION>,
) -> usize {
    let mut accumulator = 0;
    for d in 0..GRID_DIMENSION {
        debug_assert!(coord[d] >= 0);
        let mut dim_accumulator = coord[d] as usize;
        for dn in (d + 1)..GRID_DIMENSION {
            dim_accumulator *= bound[dn] as usize;
        }
        accumulator += dim_accumulator;
    }
    accumulator
}

pub fn linear_to_coord<const GRID_DIMENSION: usize>(
    linear_index: usize,
    bound: &Coord<GRID_DIMENSION>,
) -> Coord<GRID_DIMENSION> {
    let mut result = Coord::zero();
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
    index: &Coord<GRID_DIMENSION>,
    bound: &Coord<GRID_DIMENSION>,
) -> Coord<GRID_DIMENSION> {
    let mut result = Coord::zero();
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

pub fn coord_to_linear_in_box<const GRID_DIMENSION: usize>(
    coord: &Coord<GRID_DIMENSION>,
    b: &Box<GRID_DIMENSION>,
) -> usize {
    #[cfg(test)]
    {
        // TODO replace with `debug_assert!(b,contains ...)`
        for d in 0..GRID_DIMENSION {
            assert!(coord[d] >= b[(d, 0)]);
            assert!(coord[d] <= b[(d, 1)]);
        }
    }

    let bound = (b.column(1) - b.column(0)).add_scalar(1);
    let translated_coord = coord - b.column(0);
    let mut accumulator = 0;
    for d in 0..GRID_DIMENSION {
        let mut dim_accumulator = translated_coord[d] as usize;
        for dn in (d + 1)..GRID_DIMENSION {
            dim_accumulator *= bound[dn] as usize;
        }
        accumulator += dim_accumulator;
    }
    accumulator
}

pub fn linear_to_coord_in_box<const GRID_DIMENSION: usize>(
    index: usize,
    b: &Box<GRID_DIMENSION>,
) -> Coord<GRID_DIMENSION> {
    let bound = (b.column(1) - b.column(0)).add_scalar(1);
    debug_assert!(index <= real_buffer_size(&bound));

    let mut result = Coord::zero();
    let mut index_accumulator = index;
    for d in 0..GRID_DIMENSION - 1 {
        let mut dim_accumulator = 1;
        for dn in (d + 1)..GRID_DIMENSION {
            dim_accumulator *= bound[dn] as usize;
        }

        result[d] = (index_accumulator / dim_accumulator) as i32;
        index_accumulator %= dim_accumulator;
    }
    result[GRID_DIMENSION - 1] = index_accumulator as i32;
    result + b.column(0)
}

pub fn coord_in_box<const GRID_DIMENSION: usize>(
    coord: &Coord<GRID_DIMENSION>,
    view: &Box<GRID_DIMENSION>,
) -> bool {
    for d in 0..GRID_DIMENSION {
        if coord[d] < view[(d, 0)] || coord[d] > view[(d, 1)] {
            return false;
        }
    }
    true
}

/// Check whether b is contained in a.
/// Remeber, boxes are inclusive.
pub fn box_contains_box<const GRID_DIMENSION: usize>(
    a: &Box<GRID_DIMENSION>,
    b: &Box<GRID_DIMENSION>,
) -> bool {
    for d in 0..GRID_DIMENSION {
        if b[(d, 0)] < a[(d, 0)] || b[(d, 1)] > a[(d, 1)] {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use nalgebra::{matrix, vector};

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
    fn box_buffer_size_test() {
        {
            let dimensions = matrix![0, 5];
            let real_size = box_buffer_size(&dimensions);
            assert_eq!(real_size, 6);
            let complex_size = box_complex_buffer_size(&dimensions);
            assert_eq!(complex_size, (6 / 2) + 1);
        }

        {
            let dimensions = matrix![0, 5; 0, 7; 0, 9];
            let real_size = box_buffer_size(&dimensions);
            assert_eq!(real_size, 6 * 8 * 10);
            let complex_size = box_complex_buffer_size(&dimensions);
            assert_eq!(complex_size, 6 * 8 * ((10 / 2) + 1));
        }

        {
            let dimensions = matrix![1, 6; 1, 8; 1, 10];
            let real_size = box_buffer_size(&dimensions);
            assert_eq!(real_size, 6 * 8 * 10);
            let complex_size = box_complex_buffer_size(&dimensions);
            assert_eq!(complex_size, 6 * 8 * ((10 / 2) + 1));
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

    #[test]
    fn coord_to_linear_in_box_test() {
        assert_eq!(
            coord_to_linear_in_box(&vector![5, 5, 5], &matrix![0, 9; 0, 9; 0, 9]),
            linear_index(&vector![5, 5, 5], &vector![10, 10, 10])
        );

        assert_eq!(
            coord_to_linear_in_box(&vector![5, 5, 5], &matrix![2, 8; 2, 8; 2, 8]),
            linear_index(&vector![3, 3, 3], &vector![7, 7, 7])
        );
    }

    #[test]
    fn linear_to_coord_in_box_test() {
        assert_eq!(
            linear_to_coord_in_box(5, &matrix![2, 8]),
            linear_to_coord(7, &vector![10])
        );
    }

    #[test]
    fn in_box_comp_test() {
        {
            let bound = matrix![0, 9];
            let c = vector![8];
            let li = coord_to_linear_in_box(&c, &bound);
            assert_eq!(c, linear_to_coord_in_box(li, &bound));
        }

        {
            let bound = matrix![0, 9; 0, 9];
            let c = vector![9, 8];
            let li = coord_to_linear_in_box(&c, &bound);
            assert_eq!(c, linear_to_coord_in_box(li, &bound));
        }
    }

    #[test]
    fn box_contains_box_test() {
        {
            let a = matrix![1, 2];
            let b = matrix![1, 2];
            assert!(box_contains_box(&a, &b));
        }

        {
            let a = matrix![0, 9];
            let b = matrix![1, 2];
            assert!(box_contains_box(&a, &b));
        }

        {
            let a = matrix![2, 9];
            let b = matrix![1, 2];
            assert!(!box_contains_box(&a, &b));
        }

        {
            let a = matrix![2, 9];
            let b = matrix![3, 10];
            assert!(!box_contains_box(&a, &b));
        }
    }
}
