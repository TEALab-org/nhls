use crate::domain::*;
use crate::par_stencil;
use crate::stencil::*;

pub fn box_solve<
    'a,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
>(
    stencil: &StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    input: &mut Domain<'a, GRID_DIMENSION>,
    output: &mut Domain<'a, GRID_DIMENSION>,
    steps: usize,
    chunk_size: usize,
) where
    Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
{
    debug_assert_eq!(input.aabb(), output.aabb());
    for _ in 0..steps - 1 {
        {
            let bc = PeriodicCheck::new(input);
            par_stencil::apply(&bc, stencil, input, output, chunk_size);
        }
        std::mem::swap(input, output);
    }
    //println!("final t");
    let bc = PeriodicCheck::new(input);
    par_stencil::apply(&bc, stencil, input, output, chunk_size);
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::util::*;
    use fftw::array::AlignedVec;
    use float_cmp::assert_approx_eq;
    use nalgebra::matrix;

    fn test_unit_stencil<
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    >(
        stencil: &StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        bound: &AABB<GRID_DIMENSION>,
        steps: usize,
    ) where
        Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
    {
        let chunk_size = 3;
        assert_eq!(stencil.apply(&[1.0; NEIGHBORHOOD_SIZE]), 1.0);
        let n_r = bound.buffer_size();

        let mut input_buffer = vec![1.0; n_r];
        let mut output_buffer = vec![2.0; n_r];
        let mut input_domain = Domain::new(*bound, &mut input_buffer);
        let mut output_domain = Domain::new(*bound, &mut output_buffer);
        box_solve(
            stencil,
            &mut input_domain,
            &mut output_domain,
            steps,
            chunk_size,
        );

        for x in &output_buffer[0..n_r] {
            assert_approx_eq!(f32, *x, 1.0);
        }
    }

    #[test]
    fn test_1d_simple() {
        let stencil = Stencil::new([[0]], |args: &[f32; 1]| args[0]);
        let bound = AABB::new(matrix![0, 100]);
        test_unit_stencil(&stencil, &bound, 100);
    }

    #[test]
    fn test_2d_simple() {
        let stencil = Stencil::new([[0, 0]], |args: &[f32; 1]| args[0]);
        let bound = AABB::new(matrix![0, 50; 0, 50]);
        test_unit_stencil(&stencil, &bound, 9);
    }

    #[test]
    fn test_2d_less_simple() {
        let stencil = Stencil::new(
            [[0, -1], [0, 1], [1, 0], [-1, 0], [0, 0]],
            |args: &[f32; 5]| {
                debug_assert_eq!(args.len(), 5);
                let mut r = 0.0;
                for a in args {
                    r += a / 5.0;
                }
                r
            },
        );
        let bound = AABB::new(matrix![0, 50;0,  50]);
        test_unit_stencil(&stencil, &bound, 10);
    }

    #[test]
    fn test_1d_less_simple() {
        let stencil = Stencil::new([[-1], [1], [0]], |args: &[f32; 3]| {
            debug_assert_eq!(args.len(), 3);
            let mut r = 0.0;
            for a in args {
                r += a / 3.0
            }
            r
        });
        let bound = AABB::new(matrix![0, 100]);
        test_unit_stencil(&stencil, &bound, 10);
    }

    #[test]
    fn test_3d() {
        let stencil = Stencil::new(
            [
                [0, 0, -2],
                [4, 5, 3],
                [0, -1, 0],
                [0, 1, 0],
                [1, 0, 0],
                [-1, 0, 4],
                [0, 0, 0],
            ],
            |args: &[f32; 7]| {
                let mut r = 0.0;
                for a in args {
                    r += a / 7.0;
                }
                r
            },
        );
        let bound = AABB::new(matrix![0, 20;0,  20;0,  20]);
        test_unit_stencil(&stencil, &bound, 5);
    }

    #[test]
    fn shifter() {
        let stencil = Stencil::new([[-1]], |args: &[f32; 1]| args[0]);
        let bound = AABB::new(matrix![0, 9]);
        let mut input_buffer = AlignedVec::new(10);
        for i in 0..10 {
            input_buffer[i] = i as f32;
        }
        let mut input_domain = Domain::new(bound, input_buffer.as_slice_mut());

        let mut output_buffer = AlignedVec::new(10);
        let mut output_domain =
            Domain::new(bound, output_buffer.as_slice_mut());
        let chunk_size = 1;
        let n = 1;
        box_solve(
            &stencil,
            &mut input_domain,
            &mut output_domain,
            n,
            chunk_size,
        );
        for i in 0..10 {
            assert_approx_eq!(f32, output_buffer[(i + n) % 10], i as f32);
        }
    }
}
