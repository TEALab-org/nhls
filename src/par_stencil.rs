use crate::stencil::*;
use crate::util::*;
use rayon::prelude::*;

/// Modifies input buffer!!
pub fn apply<Operation, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>(
    stencil: &StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    bound: Bound<GRID_DIMENSION>,
    input: &mut [f32],
    output: &mut [f32],
    chunk_size: usize,
) where
    Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
{
    debug_assert_eq!(input.len(), output.len());
    output
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, output_chunk)| {
            let offset = i * chunk_size; 
            for i in 0..output_chunk.len() {
                let linear_index = offset + i;            
                let coord = linear_to_coord(linear_index, &bound);
                // Gather neighbor values

                // Evaluate stencil

                // Save result into output_chunk
            }
        });
}
