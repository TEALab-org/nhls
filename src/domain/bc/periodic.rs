use crate::domain::bc::BCCheck;
use crate::domain::Domain;
use crate::util::*;

/// World box is 0 origin for min, with bound as max
pub struct PeriodicCheck<'a, const GRID_DIMENSION: usize> {
    bound: Coord<GRID_DIMENSION>,
    domain: &'a Domain<'a, GRID_DIMENSION>,
}

impl<'a, const GRID_DIMENSION: usize> PeriodicCheck<'a, GRID_DIMENSION> {
    pub fn new(bound: Coord<GRID_DIMENSION>, domain: &'a Domain<'a, GRID_DIMENSION>) -> Self {
        PeriodicCheck { bound, domain }
    }
}

impl<const GRID_DIMENSION: usize> BCCheck<GRID_DIMENSION> for PeriodicCheck<'_, GRID_DIMENSION> {
    fn check(&self, coord: &Coord<GRID_DIMENSION>) -> Option<f32> {
        let p_coord = periodic_index(coord, &self.bound);
        if p_coord != *coord {
            return Some(self.domain.view(&p_coord));
        }
        None
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use nalgebra::{matrix, vector};

    #[test]
    fn periodic_lookup_test() {
        {
            let view_box = matrix![0, 10];
            let n_r = box_buffer_size(&view_box);
            let mut buffer = fftw::array::AlignedVec::new(n_r);
            for i in 0..n_r {
                buffer.as_slice_mut()[i] = i as f32;
            }
            let domain = Domain::new(view_box, buffer.as_slice_mut());
            let max = Coord::from(view_box.column(1));
            let bc = PeriodicCheck::new(max, &domain);
            for i in 0..n_r {
                let v = bc.check(&vector![i as i32]);
                assert_eq!(v, None);
            }

            {
                let v = bc.check(&vector![-1]);
                assert!(v.is_some());
                assert_approx_eq!(f32, v.unwrap(), 9.0);
            }

            {
                let v = bc.check(&vector![10]);
                assert!(v.is_some());
                assert_approx_eq!(f32, v.unwrap(), 0.0);
            }
        }
    }
}
