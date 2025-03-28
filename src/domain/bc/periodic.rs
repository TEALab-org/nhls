use crate::domain::*;
use crate::util::*;

pub struct PeriodicCheck<
    'a,
    const GRID_DIMENSION: usize,
    DomainType: DomainView<GRID_DIMENSION>,
> {
    domain: &'a DomainType,
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        DomainType: DomainView<GRID_DIMENSION>,
    > PeriodicCheck<'a, GRID_DIMENSION, DomainType>
{
    pub fn new(domain: &'a DomainType) -> Self {
        PeriodicCheck { domain }
    }
}

impl<const GRID_DIMENSION: usize, DomainType: DomainView<GRID_DIMENSION>>
    BCCheck<GRID_DIMENSION> for PeriodicCheck<'_, GRID_DIMENSION, DomainType>
{
    fn check(
        &self,
        world_coord: &Coord<GRID_DIMENSION>,
        _global_time: usize,
    ) -> Option<f64> {
        let p_coord = &self.domain.aabb().periodic_coord(world_coord);
        if p_coord != world_coord {
            return Some(self.domain.view(p_coord));
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

            let mut domain = OwnedDomain::new(aabb);
            domain.par_set_values(|coord| coord[0] as f64, 1);
            let bc = PeriodicCheck::new(&domain);
            for (i, _) in domain.buffer().iter().enumerate() {
                let v = bc.check(&vector![i as i32], 0);
                assert_eq!(v, None);
            }

            {
                let v = bc.check(&vector![-1], 1);
                assert!(v.is_some());
                assert_approx_eq!(f64, v.unwrap(), 10.0);
            }

            {
                let v = bc.check(&vector![11], 2);
                assert!(v.is_some());
                assert_approx_eq!(f64, v.unwrap(), 0.0);
            }
        }
    }
}
