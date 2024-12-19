use crate::domain::bc::BCCheck;
use crate::domain::Domain;
use crate::util::*;

pub struct PeriodicCheck<'a, const GRID_DIMENSION: usize> {
    domain: &'a Domain<'a, GRID_DIMENSION>,
}

impl<'a, const GRID_DIMENSION: usize> PeriodicCheck<'a, GRID_DIMENSION> {
    pub fn new(domain: &'a Domain<'a, GRID_DIMENSION>) -> Self {
        PeriodicCheck { domain }
    }
}

impl<const GRID_DIMENSION: usize> BCCheck<GRID_DIMENSION>
    for PeriodicCheck<'_, GRID_DIMENSION>
{
    fn check(&self, world_coord: &Coord<GRID_DIMENSION>) -> Option<f64> {
        let p_coord = &self.domain.aabb().periodic_coord(world_coord);
        if p_coord != world_coord {
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
    fn periodic_check_test() {
        {
            let aabb = AABB::new(matrix![0, 10]);
            let n_r = aabb.buffer_size();
            let mut buffer = fftw::array::AlignedVec::new(n_r);
            for i in 0..n_r {
                buffer.as_slice_mut()[i] = i as f64;
            }
            let domain = Domain::new(aabb, buffer.as_slice_mut());
            let bc = PeriodicCheck::new(&domain);
            for i in 0..n_r {
                let v = bc.check(&vector![i as i32]);
                assert_eq!(v, None);
            }

            {
                let v = bc.check(&vector![-1]);
                assert!(v.is_some());
                assert_approx_eq!(f64, v.unwrap(), 10.0);
            }

            {
                let v = bc.check(&vector![11]);
                assert!(v.is_some());
                assert_approx_eq!(f64, v.unwrap(), 0.0);
            }
        }
    }

    #[test]
    fn periodic_domain_swap() {
        let bound = AABB::new(matrix![0, 5; 0, 5]);
        let n_r = bound.buffer_size();
        let mut buffer_a = fftw::array::AlignedVec::new(n_r);
        let mut buffer_b = fftw::array::AlignedVec::new(n_r);
        for i in 0..n_r {
            buffer_a[i] = i as f64;
            buffer_b[i] = (n_r - i) as f64;
        }
    }
}
