use crate::direct_solver::*;
use crate::domain::*;
use crate::par_stencil;
use crate::stencil::*;
use crate::util::*;

/// Generic direct solver for time-invariant stencils
/// in any dimension and of any size.
/// You should prefer an optimized direct solver if available.
/// Supports arbitrary boundary conditions.
pub struct DirectFrustrumSolver<
    'a,
    BC,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    BC: BCCheck<GRID_DIMENSION>,
{
    pub bc: &'a BC,
    pub stencil: &'a Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
    pub chunk_size: usize,
}

impl<BC, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>
    DirectFrustrumSolver<'_, BC, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    BC: BCCheck<GRID_DIMENSION>,
{
    pub fn apply<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        sloped_sides: &Bounds<GRID_DIMENSION>,
        steps: usize,
        mut global_time: usize,
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
            par_stencil::apply(
                self.bc,
                self.stencil,
                input_domain,
                output_domain,
                global_time,
                self.chunk_size,
            );
            std::mem::swap(input_domain, output_domain);
        }
        std::mem::swap(input_domain, output_domain);
    }
}

impl<
        BC: BCCheck<GRID_DIMENSION>,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    > DirectSolverInterface<GRID_DIMENSION>
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
