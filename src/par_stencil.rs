use crate::domain::*;
use crate::stencil::*;
use crate::util::*;
use rayon::prelude::*;

pub fn box_apply<BC, Operation, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>(
    bc: &BC,
    stencil: &StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    input: &Domain<GRID_DIMENSION>,
    output: &mut Domain<GRID_DIMENSION>,
    chunk_size: usize,
) where
    Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    output
        .par_modify_access(chunk_size)
        .for_each(|mut d: DomainChunk<'_, GRID_DIMENSION>| {
            d.coord_iter_mut().for_each(
                |(world_coord, value_mut): (Coord<GRID_DIMENSION>, &mut f32)| {
                    let args = gather_args(bc, stencil, input, &world_coord);
                    let result = stencil.apply(&args);
                    *value_mut = result;
                },
            )
        })
}

/*
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
            box_apply(&lookup, &input, &stencil, &max_size, &mut output, 1);
            for x in &output {
                assert_approx_eq!(f32, *x, 1.0);
            }
        }

        {
            let input = vec![2.0; n_r];
            let lookup = PeriodicBCLookup::new(max_size);
            let mut output = vec![0.0; n_r];
            box_apply(&lookup, &input, &stencil, &max_size, &mut output, 3);
            for x in &output {
                assert_approx_eq!(f32, *x, 2.0);
            }
        }
    }
}
*/
