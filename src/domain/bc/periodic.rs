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

impl<const GRID_DIMENSION: usize> BCCheck<GRID_DIMENSION> for PeriodicCheck<'_, GRID_DIMENSION> {
    fn check(&self, world_coord: &Coord<GRID_DIMENSION>) -> Option<f32> {
        let p_coord = periodic_coord(world_coord, &self.domain.view_box());
        if p_coord != *world_coord {
            return Some(self.domain.view(&p_coord));
        }
        None
    }
}

/// Find the coord within bound assuming periodic boundary conditions.
/// Assumes that coords are no more than one box away!
fn periodic_coord<const GRID_DIMENSION: usize>(
    index: &Coord<GRID_DIMENSION>,
    bound: &Box<GRID_DIMENSION>,
) -> Coord<GRID_DIMENSION> {
    let mut result = Coord::zero();
    for d in 0..GRID_DIMENSION {
        let di_raw = index[d];
        result[d] = if di_raw < bound[(d, 0)] {
            bound[(d, 1)] + 1 + di_raw
        } else if di_raw > bound[(d, 1)] {
            bound[(d, 0)] + (di_raw - bound[(d, 1)] - 1)
        } else {
            di_raw
        }
    }
    result
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use nalgebra::{matrix, vector};

    #[test]
    fn periodic_check_test() {
        {
            let view_box = matrix![0, 10];
            let n_r = box_buffer_size(&view_box);
            let mut buffer = fftw::array::AlignedVec::new(n_r);
            for i in 0..n_r {
                buffer.as_slice_mut()[i] = i as f32;
            }
            let domain = Domain::new(view_box, buffer.as_slice_mut());
            let bc = PeriodicCheck::new(&domain);
            for i in 0..n_r {
                let v = bc.check(&vector![i as i32]);
                assert_eq!(v, None);
            }

            {
                let v = bc.check(&vector![-1]);
                assert!(v.is_some());
                assert_approx_eq!(f32, v.unwrap(), 10.0);
            }

            {
                let v = bc.check(&vector![11]);
                assert!(v.is_some());
                assert_approx_eq!(f32, v.unwrap(), 0.0);
            }
        }
    }

    #[test]
    fn periodic_coord_test() {
        {
            let index = vector![0, 0];
            let bound = matrix![0, 10; 0, 10];
            assert_eq!(periodic_coord(&index, &bound), vector![0, 0]);
        }

        {
            let index = vector![10, 10];
            let bound = matrix![0, 10; 0, 10];
            assert_eq!(periodic_coord(&index, &bound), vector![10, 10]);
        }

        {
            let index = vector![-1, 0];
            let bound = matrix![0, 10; 0, 10];
            assert_eq!(periodic_coord(&index, &bound), vector![10, 0]);
        }

        {
            let index = vector![0, -1];
            let bound = matrix![0, 10; 0, 10];
            assert_eq!(periodic_coord(&index, &bound), vector![0, 10]);
        }

        {
            let index = vector![0, -1, -4, -19, 134];
            let bound = matrix![0, 100; 0, 100;0, 100; 0, 100;0, 100];
            assert_eq!(periodic_coord(&index, &bound), vector![0, 100, 97, 82, 33]);
        }
    }

    #[test]
    fn periodic_domain_swap() {
        let bound = matrix![0, 5; 0, 5];
        let n_r = box_buffer_size(&bound);
        let mut buffer_a = fftw::array::AlignedVec::new(n_r);
        let mut buffer_b = fftw::array::AlignedVec::new(n_r);
        for i in 0..n_r {
            buffer_a[i] = i as f32;
            buffer_b[i] = (n_r - i) as f32;
        }
    }
}
