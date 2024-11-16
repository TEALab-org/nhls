use crate::domain::*;
use crate::par_stencil;
use crate::stencil::*;
use crate::util::*;
use fftw::array::*;

pub fn box_apply<'a, BC, Operation, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>(
    bc: &BC,
    stencil: &StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    input: &mut Domain<'a, GRID_DIMENSION>,
    output: &mut Domain<'a, GRID_DIMENSION>,
    steps: usize,
    chunk_size: usize,
) where
    Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    debug_assert_eq!(input.view_box(), output.view_box());
    for _ in 0..steps - 1 {
        par_stencil::box_apply(bc, stencil, input, output, chunk_size);
        std::mem::swap(input, output);
    }
    par_stencil::box_apply(bc, stencil, input, output, chunk_size);
}
/*

#[cfg(test)]
mod unit_tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use nalgebra::vector;

    fn test_unit_stencil<
        Lookup,
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    >(
        stencil: &StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        bc_lookup: &Lookup,
        bound: Coord<GRID_DIMENSION>,
        n: usize,
    ) where
        Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
        Lookup: BCLookup<GRID_DIMENSION>,
    {
        let chunk_size = 3;
        assert_eq!(stencil.apply(&[1.0; NEIGHBORHOOD_SIZE]), 1.0);
        let rbs = real_buffer_size(&bound);

        let mut input_x = fftw::array::AlignedVec::new(rbs);
        for x in input_x.as_slice_mut() {
            *x = 1.0f32;
        }
        let mut result_buffer = fftw::array::AlignedVec::new(rbs);
        naive_block_solve(
            bc_lookup,
            stencil,
            bound,
            n,
            &mut input_x,
            &mut result_buffer,
            chunk_size,
        );

        for x in &result_buffer[0..rbs] {
            assert_approx_eq!(f32, *x, 1.0);
        }
    }

    #[test]
    fn test_1d_simple() {
        let stencil = Stencil::new([[0]], |args: &[f32; 1]| args[0]);
        let max_size = vector![100];
        let lookup = PeriodicBCLookup::new(max_size);
        test_unit_stencil(&stencil, &lookup, max_size, 100);
    }

    #[test]
    fn test_2d_simple() {
        let stencil = Stencil::new([[0, 0]], |args: &[f32; 1]| args[0]);
        let max_size = vector![50, 50];
        let lookup = PeriodicBCLookup::new(max_size);
        test_unit_stencil(&stencil, &lookup, max_size, 9);
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
        let max_size = vector![50, 50];
        let lookup = PeriodicBCLookup::new(max_size);
        test_unit_stencil(&stencil, &lookup, max_size, 10);
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
        let max_size = vector![100];
        let lookup = PeriodicBCLookup::new(max_size);
        test_unit_stencil(&stencil, &lookup, max_size, 10);
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
        {
            let max_size = vector![20, 20, 20];
            let lookup = ConstantBCLookup::new(1.0, max_size);
            test_unit_stencil(&stencil, &lookup, max_size, 5);
        }

        {
            let max_size = vector![11, 9, 20];
            let lookup = ConstantBCLookup::new(1.0, max_size);
            test_unit_stencil(&stencil, &lookup, max_size, 5);
        }
    }

    #[test]
    fn shifter() {
        let stencil = Stencil::new([[-1]], |args: &[f32; 1]| args[0]);
        let max_size = vector![10];
        let mut input_x = AlignedVec::new(10);
        for i in 0..10 {
            input_x[i] = i as f32;
        }
        let mut output_x = AlignedVec::new(10);
        let bc_lookup = PeriodicBCLookup::new(max_size);
        let chunk_size = 1;
        let n = 3;
        naive_block_solve(
            &bc_lookup,
            &stencil,
            max_size,
            n,
            &mut input_x,
            &mut output_x,
            chunk_size,
        );
        println!("output: {:?}", output_x.as_slice());
        for i in 0..10 {
            assert_approx_eq!(f32, output_x[(i + n) % 10], i as f32);
        }
    }
}
*/
