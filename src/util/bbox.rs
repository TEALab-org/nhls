use crate::util::indexing::*;
use crate::util::*;

type Bounds<const DIMENSION: usize> = nalgebra::SMatrix<i32, { DIMENSION }, 2>;

pub struct AABB<const DIMENSION: usize> {
    bounds: Bounds<DIMENSION>,
}

impl<const DIMENSION: usize> AABB<DIMENSION> {
    pub fn new(bounds: Bounds<DIMENSION>) -> Self {
        AABB { bounds }
    }

    pub fn from_mm(min: Coord<DIMENSION>, max: Coord<DIMENSION>) -> Self {
        for d in 0..DIMENSION {
            debug_assert!(min[d] <= max[d])
        }

        AABB {
            bounds: Bounds::from_columns(&[min, max]),
        }
    }

    /// Moving min to the origin, what is the size in each direction, exclusive, 0..exclusive_max
    pub fn exclusive_bounds(&self) -> Coord<DIMENSION> {
        (self.bounds.column(1) - self.bounds.column(0)).add_scalar(1)
    }

    pub fn buffer_size(&self) -> usize {
        real_buffer_size(&self.exclusive_bounds())
    }

    pub fn complex_buffer_size(&self) -> usize {
        complex_buffer_size(&self.exclusive_bounds())
    }

    pub fn linear_index(&self, coord: &Coord<DIMENSION>) -> usize {
        let mut accumulator = 0;
        for d in 0..DIMENSION {
            debug_assert!(coord[d] >= 0);
            let mut dim_accumulator = coord[d] as usize;
            for dn in (d + 1)..DIMENSION {
                dim_accumulator *= self.bounds[dn] as usize;
            }
            accumulator += dim_accumulator;
        }
        accumulator
    }

    pub fn coord_to_linear(&self, coord: &Coord<DIMENSION>) -> usize {
        coord_to_linear(coord, &self.exclusive_bounds())
    }

    pub fn linear_to_coord(&self, index: usize) -> Coord<DIMENSION> {
        linear_to_coord(index, &self.exclusive_bounds())
    }

    pub fn contains(&self, coord: &Coord<DIMENSION>) -> bool {
        for d in 0..DIMENSION {
            if coord[d] < self.bounds[(d, 0)] || coord[d] > self.bounds[(d, 1)] {
                return false;
            }
        }
        true
    }

    pub fn contains_aabb(&self, other: &Self) -> bool {
        for d in 0..DIMENSION {
            if other.bounds[(d, 0)] <= self.bounds[(d, 0)]
                || other.bounds[(d, 1)] > self.bounds[(d, 1)]
            {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use nalgebra::{matrix, vector};

    #[test]
    fn buffer_size_test() {
        {
            let a = AABB::new(matrix![0, 5]);
            let real_size = (&dimensions);
            assert_eq!(real_size, 6);
            let complex_size = box_complex_buffer_size(&dimensions);
            assert_eq!(complex_size, (6 / 2) + 1);
        }

        {
            let dimensions = matrix![0, 5; 0, 7; 0, 9];
            let real_size = dimensions.buffer_size();
            assert_eq!(real_size, 6 * 8 * 10);
            let complex_size = box_complex_buffer_size(&dimensions);
            assert_eq!(complex_size, 6 * 8 * ((10 / 2) + 1));
        }

        {
            let dimensions = matrix![1, 6; 1, 8; 1, 10];
            let real_size = dimensions.buffer_size();
            assert_eq!(real_size, 6 * 8 * 10);
            let complex_size = box_complex_buffer_size(&dimensions);
            assert_eq!(complex_size, 6 * 8 * ((10 / 2) + 1));
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
}
