use crate::domain::bc::BCCheck;
use crate::util::*;

pub struct ConstantCheck<const GRID_DIMENSION: usize> {
    value: f32,
    bound: Coord<GRID_DIMENSION>,
}

impl<const GRID_DIMENSION: usize> ConstantCheck<GRID_DIMENSION> {
    pub fn new(value: f32, bound: Coord<GRID_DIMENSION>) -> Self {
        ConstantCheck { value, bound }
    }
}

impl<const GRID_DIMENSION: usize> BCCheck<GRID_DIMENSION> for ConstantCheck<GRID_DIMENSION> {
    fn check(&self, coord: &Coord<GRID_DIMENSION>) -> Option<f32> {
        for d in 0..GRID_DIMENSION {
            let c = coord[d];
            if c < 0 || c >= self.bound[d] {
                return Some(self.value);
            }
        }
        None
    }
}

/*
#[cfg(test)]
mod unit_tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use nalgebra::vector;

    #[test]
    fn constant_lookup_test() {
        let bound = vector![10];
        let n_r = real_buffer_size(&bound);
        let mut buffer = fftw::array::AlignedVec::new(n_r);
        for i in 0..n_r {
            buffer.as_slice_mut()[i] = i as f32;
        }
        let lookup = ConstantBCLookup::new(-1.0, bound);
        for i in 0..n_r {
            let v = lookup.value(&vector![i as i32], &buffer);
            let e = i as f32;
            assert_approx_eq!(f32, v, e);
        }

        {
            let v = lookup.value(&vector![-1], &buffer);
            let e = -1.0;
            assert_approx_eq!(f32, v, e);
        }

        {
            let v = lookup.value(&vector![10], &buffer);
            let e = -1.0;
            assert_approx_eq!(f32, v, e);
        }
    }
*/
