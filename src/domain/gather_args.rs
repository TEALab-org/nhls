use crate::domain::*;
use crate::stencil::*;
use crate::util::*;

pub fn gather_args<
    BC,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    DomainType: DomainView<GRID_DIMENSION>,
>(
    stencil: &Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    bc: &BC,
    input: &DomainType,
    world_coord: &Coord<GRID_DIMENSION>,
    global_time: usize,
) -> Values<NEIGHBORHOOD_SIZE>
where
    BC: BCCheck<GRID_DIMENSION>,
{
    let mut result = Values::zero();
    for (i, n_i) in stencil.offsets().iter().enumerate() {
        let n_world_coord = world_coord + n_i;
        result[i] = bc
            .check(&n_world_coord, global_time)
            .unwrap_or_else(|| input.view(&n_world_coord));
    }
    result
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use nalgebra::{matrix, vector};

    #[test]
    fn gather_args_test_const() {
        let bound = AABB::new(matrix![0, 9; 0, 9]);
        let mut domain = OwnedDomain::new(bound);
        domain.par_set_values(
            |coord: Coord<2>| (coord[0] + 3 * coord[1]) as f64,
            1,
        );
        let bc = ConstantCheck::new(-4.0, bound);
        let stencil = Stencil::new(
            [[0, -1], [0, 1], [1, 0], [-1, 0], [0, 0]],
            |_: &[f64; 5]| -1.0,
        );
        let r = gather_args(&stencil, &bc, &domain, &vector![9, 9], 0);
        let e = [
            (9 + 3 * 8) as f64,
            -4.0,
            -4.0,
            (8 + 3 * 9) as f64,
            (9 + 3 * 9) as f64,
        ];
        for n in 0..r.len() {
            assert_approx_eq!(f64, r[n], e[n]);
        }
    }

    #[test]
    fn gather_args_test_periodic() {
        let bound = AABB::new(matrix![0, 9; 0, 9]);
        let n_r = bound.buffer_size();
        let mut buffer = fftw::array::AlignedVec::new(n_r);
        for i in 0..n_r {
            let coord = bound.linear_to_coord(i);
            buffer.as_slice_mut()[i] = (coord[0] + 3 * coord[1]) as f64;
        }
        let mut domain = OwnedDomain::new(bound);
        domain.par_set_values(|coord| (coord[0] + 3 * coord[1]) as f64, 1);
        let bc = PeriodicCheck::new(&domain);
        let stencil = Stencil::new(
            [[0, -1], [0, 1], [1, 0], [-1, 0], [0, 0]],
            |_: &[f64; 5]| -1.0,
        );
        let r = gather_args(&stencil, &bc, &domain, &vector![9, 9], 11);
        let e = [
            (9 + 3 * 8) as f64,
            9.0,
            (3 * 9) as f64,
            (8 + 3 * 9) as f64,
            (9 + 3 * 9) as f64,
        ];
        for n in 0..r.len() {
            assert_approx_eq!(f64, r[n], e[n]);
        }
    }
}
