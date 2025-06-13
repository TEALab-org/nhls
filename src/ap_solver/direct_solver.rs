use crate::domain::*;
use crate::fft_solver::DirectFrustrumSolver;
use crate::stencil::*;
use crate::time_varying::tv_direct_frustrum_solver::*;
use crate::util::*;
use rayon::prelude::*;

pub trait DirectSolver<const GRID_DIMENSION: usize>: Send + Sync {
    fn apply<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        sloped_sides: &Bounds<GRID_DIMENSION>,
        steps: usize,
        global_time: usize,
        threads: usize,
    );
}

impl<
        BC: BCCheck<GRID_DIMENSION>,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    > DirectSolver<GRID_DIMENSION>
    for DirectFrustrumSolver<'_, BC, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
{
    fn apply<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        sloped_sides: &Bounds<GRID_DIMENSION>,
        steps: usize,
        global_time: usize,
        _threads: usize,
    ) {
        self.apply(
            input_domain,
            output_domain,
            sloped_sides,
            steps,
            global_time,
        );
    }
}

impl<
        'a,
        BC: BCCheck<GRID_DIMENSION>,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    > DirectSolver<GRID_DIMENSION>
    for TVDirectFrustrumSolver<
        'a,
        BC,
        GRID_DIMENSION,
        NEIGHBORHOOD_SIZE,
        StencilType,
    >
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
