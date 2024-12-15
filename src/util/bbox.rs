use crate::util::indexing::*;
use crate::util::*;

pub type Bounds<const DIMENSION: usize> = nalgebra::SMatrix<i32, { DIMENSION }, 2>;

#[derive(Hash, Debug, Copy, Clone, Eq, PartialEq)]
pub struct AABB<const DIMENSION: usize> {
    pub bounds: Bounds<DIMENSION>,
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

    pub fn add_bounds_diff(&self, diff: Bounds<DIMENSION>) -> Self {
        Self::new(self.bounds + diff)
    }

    /// Find the coord within bound assuming periodic boundary conditions.
    /// Assumes that coords are no more than one box away!
    pub fn periodic_coord(&self, coord: &Coord<DIMENSION>) -> Coord<DIMENSION> {
        let mut result = Coord::zero();
        for d in 0..DIMENSION {
            let di_raw = coord[d];
            result[d] = if di_raw < self.bounds[(d, 0)] {
                self.bounds[(d, 1)] + 1 + di_raw
            } else if di_raw > self.bounds[(d, 1)] {
                self.bounds[(d, 0)] + (di_raw - self.bounds[(d, 1)] - 1)
            } else {
                di_raw
            }
        }
        //println!("periodic_coord, c: {:?}, r: {:?}", index, result);
        result
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
            let real_size = a.buffer_size();
            assert_eq!(real_size, 6);
            let complex_size = a.complex_buffer_size();
            assert_eq!(complex_size, (6 / 2) + 1);
        }

        {
            let dimensions = AABB::new(matrix![0, 5; 0, 7; 0, 9]);
            let real_size = dimensions.buffer_size();
            assert_eq!(real_size, 6 * 8 * 10);
            let complex_size = dimensions.complex_buffer_size();
            assert_eq!(complex_size, 6 * 8 * ((10 / 2) + 1));
        }

        {
            let dimensions = AABB::new(matrix![1, 6; 1, 8; 1, 10]);
            let real_size = dimensions.buffer_size();
            assert_eq!(real_size, 6 * 8 * 10);
            let complex_size = dimensions.complex_buffer_size();
            assert_eq!(complex_size, 6 * 8 * ((10 / 2) + 1));
        }
    }

    #[test]
    fn coord_to_linear_in_box_test() {
        let bb = AABB::new(matrix![0, 9; 0, 9; 0, 9]);
        let lin_1 = bb.coord_to_linear(&vector![5, 5, 5]);
        let lin_2 = coord_to_linear(&vector![5, 5, 5], &vector![10, 10, 10]);
        assert_eq!(lin_1, lin_2);
    }

    #[test]
    fn linear_to_coord_in_box_test() {
        let bb = AABB::new(matrix![2, 8]);
        let c_1 = bb.linear_to_coord(5);
        let c_2 = linear_to_coord(7, &vector![10]);
        assert_eq!(c_1, c_2);
    }

    #[test]
    fn in_box_comp_test() {
        {
            let bound = AABB::new(matrix![0, 9]);
            let c = vector![8];
            let li = bound.coord_to_linear(&c);
            assert_eq!(c, bound.linear_to_coord(li));
        }

        {
            let bound = AABB::new(matrix![0, 9; 0, 9]);
            let c = vector![9, 8];
            let li = bound.coord_to_linear(&c);
            assert_eq!(c, bound.linear_to_coord(li));
        }
    }

    #[test]
    fn periodic_coord_test() {
        {
            let index = vector![0, 0];
            let bound = AABB::new(matrix![0, 10; 0, 10]);
            assert_eq!(bound.periodic_coord(&index), vector![0, 0]);
        }

        {
            let index = vector![10, 10];
            let bound = AABB::new(matrix![0, 10; 0, 10]);
            assert_eq!(bound.periodic_coord(&index), vector![10, 10]);
        }

        {
            let index = vector![-1, 0];
            let bound = AABB::new(matrix![0, 10; 0, 10]);
            assert_eq!(bound.periodic_coord(&index), vector![10, 0]);
        }

        {
            let index = vector![0, -1];
            let bound = AABB::new(matrix![0, 10; 0, 10]);
            assert_eq!(bound.periodic_coord(&index), vector![0, 10]);
        }

        {
            let index = vector![0, -1, -4, -19, 134];
            let bound = AABB::new(matrix![0, 100; 0, 100;0, 100; 0, 100;0, 100]);
            assert_eq!(bound.periodic_coord(&index), vector![0, 100, 97, 82, 33]);
        }
    }
}
