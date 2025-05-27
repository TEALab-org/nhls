use crate::domain::*;
use crate::stencil::TVStencil;
use crate::util::*;
use rayon::prelude::*;

pub fn tv_par_apply<
    BC,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    DomainType: DomainView<GRID_DIMENSION>,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
>(
    bc: &BC,
    stencil: &StencilType,
    input: &DomainType,
    output: &mut DomainType,
    global_time: usize,
    chunk_size: usize,
) where
    BC: BCCheck<GRID_DIMENSION>,
{
    debug_assert!(input.aabb().contains_aabb(output.aabb()));
    output.par_modify_access(chunk_size).for_each(
        |mut d: DomainChunk<'_, GRID_DIMENSION>| {
            d.coord_iter_mut().for_each(
                |(world_coord, value_mut): (
                    Coord<GRID_DIMENSION>,
                    &mut f64,
                )| {
                    let args = gather_args(
                        stencil,
                        bc,
                        input,
                        &world_coord,
                        global_time,
                    );
                    let result = stencil.apply(&args, global_time);
                    *value_mut = result;
                },
            )
        },
    )
}

pub fn tv_box_apply<
    BC,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    DomainType: DomainView<GRID_DIMENSION>,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
>(
    bc: &BC,
    stencil: &StencilType,
    input: &mut DomainType,
    output: &mut DomainType,
    steps: usize,
    mut global_time: usize,
    chunk_size: usize,
) where
    BC: BCCheck<GRID_DIMENSION>,
{
    debug_assert_eq!(input.aabb(), output.aabb());
    for _ in 0..steps - 1 {
        global_time += 1;
        tv_par_apply(bc, stencil, input, output, global_time, chunk_size);
        std::mem::swap(input, output);
    }
    global_time += 1;
    tv_par_apply(bc, stencil, input, output, global_time, chunk_size);
}
