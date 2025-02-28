use crate::domain::*;
use crate::time_varying::*;
use crate::util::*;
use rayon::prelude::*;

// Used to direct solve frustrum regions.
pub struct TVDirectFrustrumSolver<
    'a,
    BC,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> where
    BC: BCCheck<GRID_DIMENSION>,
{
    pub bc: &'a BC,
    pub stencil: &'a StencilType,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
    pub chunk_size: usize,
}

impl<
        BC,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    > TVDirectSolver<GRID_DIMENSION>
    for TVDirectFrustrumSolver<
        '_,
        BC,
        GRID_DIMENSION,
        NEIGHBORHOOD_SIZE,
        StencilType,
    >
where
    BC: BCCheck<GRID_DIMENSION>,
{
    fn apply<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        sloped_sides: &Bounds<GRID_DIMENSION>,
        steps: usize,
        mut global_time: usize,
        _threads: usize,
    ) {
        assert_eq!(input_domain.aabb(), output_domain.aabb());

        let mut trapezoid_slopes =
            self.stencil_slopes.component_mul(sloped_sides);
        let negative_slopes = -1 * trapezoid_slopes.column(1);
        trapezoid_slopes.set_column(1, &negative_slopes);

        let mut output_box = *input_domain.aabb();
        for _ in 0..steps {
            global_time += 1;
            output_box = output_box.add_bounds_diff(trapezoid_slopes);
            debug_assert!(
                input_domain.aabb().buffer_size() >= output_box.buffer_size()
            );
            output_domain.set_aabb(output_box);
            output_domain.par_modify_access(self.chunk_size).for_each(
                |mut d: DomainChunk<'_, GRID_DIMENSION>| {
                    d.coord_iter_mut().for_each(
                        |(world_coord, value_mut): (
                            Coord<GRID_DIMENSION>,
                            &mut f64,
                        )| {
                            let args = gather_args(
                                self.stencil,
                                self.bc,
                                input_domain,
                                &world_coord,
                                global_time,
                            );
                            let result = self.stencil.apply(&args, global_time);
                            *value_mut = result;
                        },
                    )
                },
            );

            std::mem::swap(input_domain, output_domain);
        }
        std::mem::swap(input_domain, output_domain);
    }
}
