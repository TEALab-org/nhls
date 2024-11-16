/*
use crate::boundary::*;
use crate::stencil::*;
use crate::util::*;
use fftw::array::*;

pub fn naive_trapezoid_solve<
    Lookup,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
>(
    bc_lookup: &Lookup,
    stencil: &StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    start_box: &Box<GRID_DIMENSION>,
    sloped_sides: Box<GRID_DIMENSION>,
    n: usize,
    input: &mut AlignedVec<f32>,
    output: &mut AlignedVec<f32>,
    chunk_size: usize,
) -> Box<GRID_DIMENSION>
where
    Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
    Lookup: BCLookup<GRID_DIMENSION>,
{
    debug_assert_eq!(
        input.len(),
        real_buffer_size(&(start_box.column(0) - start_box.column(1)))
    );
    debug_assert_eq!(
        output.len(),
        real_buffer_size(&(start_box.column(0) - start_box.column(1)))
    );

    let mut trapezoid_slopes = stencil.slopes().component_mul(&sloped_sides);
    let negative_slopes = -1 * trapezoid_slopes.column(1);
    trapezoid_slopes.set_column(1, &negative_slopes);

    let mut input_box = start_box.clone();
    let mut output_box = start_box.clone();
    for _ in 0..n - 1 {
        input_box = output_box;
        output_box = output_box + trapezoid_slopes;

        // Par apply over
    }

    output_box
}
*/
