use crate::stencil::*;
use crate::util::*;

/// Modifies input buffer!!
pub fn naive_block_solve<Operation, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>(
    stencil: &StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    bound: Bound<GRID_DIMENSION>,
    n: usize,
    input_buffer: &mut [f32],
    output: &mut [f32],
    chunk_size: usize,
) where
    Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
{
    for _ in 0..n {}
}
