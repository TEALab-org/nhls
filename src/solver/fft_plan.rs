use fftw::plan::*;
use fftw::types::*;

use crate::util::*;

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
    index: Bound<GRID_DIMENSION>,
    bound: Bound<GRID_DIMENSION>,
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

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct FFTPlanDescriptor<const GRID_DIMENSION: usize> {
    pub space_size: Bound<GRID_DIMENSION>,
}

impl<const GRID_DIMENSION: usize> FFTPlanDescriptor<GRID_DIMENSION> {
    pub fn new(space_size: Bound<GRID_DIMENSION>) -> Self {
        FFTPlanDescriptor { space_size }
    }
}

pub struct FFTPlan {
    pub forward_plan: fftw::plan::Plan<f32, c32, fftw::plan::Plan32>,
    pub backward_plan: fftw::plan::Plan<c32, f32, fftw::plan::Plan32>,
}

impl FFTPlan {
    pub fn new<const GRID_DIMENSION: usize>(space_size: Bound<GRID_DIMENSION>) -> Self {
        let plan_size = space_size.try_cast::<usize>().unwrap();
        let forward_plan =
            fftw::plan::R2CPlan32::aligned(plan_size.as_slice(), fftw::types::Flag::ESTIMATE)
                .unwrap();
        let backward_plan =
            fftw::plan::C2RPlan32::aligned(plan_size.as_slice(), fftw::types::Flag::ESTIMATE)
                .unwrap();
        FFTPlan {
            forward_plan,
            backward_plan,
        }
    }
}

// We need storage for plans
pub struct FFTPlanLibrary<const GRID_DIMENSION: usize> {
    pub plan_map: std::collections::HashMap<FFTPlanDescriptor<GRID_DIMENSION>, FFTPlan>,
}

impl<const GRID_DIMENSION: usize> FFTPlanLibrary<GRID_DIMENSION> {
    pub fn new() -> Self {
        FFTPlanLibrary {
            plan_map: std::collections::HashMap::new(),
        }
    }

    pub fn get_plan(&mut self, size: Bound<GRID_DIMENSION>) -> &mut FFTPlan {
        let key = FFTPlanDescriptor::new(size);
        self.plan_map.entry(key).or_insert(FFTPlan::new(size))
    }
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
            assert_eq!(linear_index(index, bound), 5 * 20 * 20 + 7 * 20 + 11);
        }

        {
            let index = vector![5, 7];
            let bound = vector![20, 20];
            assert_eq!(linear_index(index, bound), 5 * 20 + 7);
        }

        {
            let index = vector![5];
            let bound = vector![20];
            assert_eq!(linear_index(index, bound), 5);
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
}
