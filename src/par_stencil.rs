use crate::boundary::*;
use crate::stencil::*;
use crate::util::*;
use rayon::prelude::*;

/// Modifies input buffer!!
pub fn apply<Lookup, Operation, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>(
    bc_lookup: &Lookup,
    input: &[f32],
    stencil: &StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    bound: &Bound<GRID_DIMENSION>,
    output: &mut [f32],
    chunk_size: usize,
) where
    Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
    Lookup: BCLookup<GRID_DIMENSION>,
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
                let args = gather_args(stencil, bc_lookup, input, &coord);

                // Evaluate stencil
                let v = stencil.apply(&args);

                // Save result into output_chunk
                output_chunk[i] = v;
            }
        });
}

#[cfg(test)]
mod unit_test {
    use super::*;
    use float_cmp::assert_approx_eq;
    use nalgebra::vector;

    #[test]
    fn par_stencil_test_1d_simple() {
        let stencil = Stencil::new([[0]], |args: &[f32; 1]| args[0]);
        let max_size = vector![100];
        let n_r = real_buffer_size(&max_size);

        {
            let input = vec![1.0; n_r];
            let lookup = PeriodicBCLookup::new(max_size);
            let mut output = vec![0.0; n_r];
            apply(&lookup, &input, &stencil, &max_size, &mut output, 1);
            for x in &output {
                assert_approx_eq!(f32, *x, 1.0);
            }
        }

        {
            let input = vec![2.0; n_r];
            let lookup = PeriodicBCLookup::new(max_size);
            let mut output = vec![0.0; n_r];
            apply(&lookup, &input, &stencil, &max_size, &mut output, 3);
            for x in &output {
                assert_approx_eq!(f32, *x, 2.0);
            }
        }
    }
}
