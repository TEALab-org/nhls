use crate::boundary::*;
use crate::par_stencil;
use crate::stencil::*;
use crate::util::*;
use fftw::array::*;

/// Modifies input buffer!!
pub fn naive_block_solve<
    Lookup,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
>(
    bc_lookup: &Lookup,
    stencil: &StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    bound: Bound<GRID_DIMENSION>,
    n: usize,
    input: &mut AlignedVec<f32>,
    output: &mut AlignedVec<f32>,
    chunk_size: usize,
) where
    Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
    Lookup: BCLookup<GRID_DIMENSION>,
{
    for _ in 0..n - 1 {
        par_stencil::apply(bc_lookup, input, stencil, &bound, output, chunk_size);
        std::mem::swap(input, output);
    }
    par_stencil::apply(bc_lookup, input, stencil, &bound, output, chunk_size);
}
