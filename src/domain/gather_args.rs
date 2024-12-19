use crate::domain::*;
use crate::stencil::*;
use crate::util::*;

pub fn gather_args<
    BC,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
>(
    stencil: &StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    bc: &BC,
    input: &Domain<GRID_DIMENSION>,
    world_coord: &Coord<GRID_DIMENSION>,
) -> [f32; NEIGHBORHOOD_SIZE]
where
    Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    let mut result = [0.0; NEIGHBORHOOD_SIZE];
    for (i, n_i) in stencil.offsets().iter().enumerate() {
        let n_world_coord = world_coord + n_i;
        result[i] = bc
            .check(&n_world_coord)
            .unwrap_or_else(|| input.view(&n_world_coord));
        /*
        println!(
            "wc: {:?}, ni: {:?}, nwc: {:?}, r: {}\n",
            world_coord, n_i, n_world_coord, result[i]
        );
        */
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
        let n_r = bound.buffer_size();
        let mut buffer = fftw::array::AlignedVec::new(n_r);
        for i in 0..n_r {
            let coord = bound.linear_to_coord(i);
            buffer.as_slice_mut()[i] = (coord[0] + 3 * coord[1]) as f32;
        }
        let domain = Domain::new(bound, buffer.as_slice_mut());
        let bc = ConstantCheck::new(-4.0, bound);
        let stencil = Stencil::new(
            [[0, -1], [0, 1], [1, 0], [-1, 0], [0, 0]],
            |_: &[f32; 5]| -1.0,
        );
        let r = gather_args(&stencil, &bc, &domain, &vector![9, 9]);
        let e = [
            (9 + 3 * 8) as f32,
            -4.0,
            -4.0,
            (8 + 3 * 9) as f32,
            (9 + 3 * 9) as f32,
        ];
        for n in 0..r.len() {
            assert_approx_eq!(f32, r[n], e[n]);
        }
    }

    #[test]
    fn gather_args_test_periodic() {
        let bound = AABB::new(matrix![0, 9; 0, 9]);
        let n_r = bound.buffer_size();
        let mut buffer = fftw::array::AlignedVec::new(n_r);
        for i in 0..n_r {
            let coord = bound.linear_to_coord(i);
            buffer.as_slice_mut()[i] = (coord[0] + 3 * coord[1]) as f32;
            println!(
                "i: {}, c: {:?}, r: {}",
                i,
                coord,
                buffer.as_slice_mut()[i]
            );
        }
        let domain = Domain::new(bound, buffer.as_slice_mut());
        let bc = PeriodicCheck::new(&domain);
        let stencil = Stencil::new(
            [[0, -1], [0, 1], [1, 0], [-1, 0], [0, 0]],
            |_: &[f32; 5]| -1.0,
        );
        let r = gather_args(&stencil, &bc, &domain, &vector![9, 9]);
        let e = [
            (9 + 3 * 8) as f32,
            9.0,
            (3 * 9) as f32,
            (8 + 3 * 9) as f32,
            (9 + 3 * 9) as f32,
        ];
        println!("r: {:?}, e: {:?}", r, e);
        for n in 0..r.len() {
            assert_approx_eq!(f32, r[n], e[n]);
        }
    }
}
