use crate::domain::*;
//use crate::fft_solver::*;
use crate::par_stencil;
use crate::stencil::*;
use crate::util::*;

pub struct DirectFrustrumSolver<
    'a,
    BC,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    pub bc: &'a BC,
    pub stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
    pub chunk_size: usize,
}

impl<
        'a,
        BC,
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    > DirectFrustrumSolver<'a, BC, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    pub fn apply<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        sloped_sides: &Bounds<GRID_DIMENSION>,
        steps: usize,
    ) {
        assert_eq!(input_domain.aabb(), output_domain.aabb());

        let mut trapezoid_slopes =
            self.stencil_slopes.component_mul(sloped_sides);
        let negative_slopes = -1 * trapezoid_slopes.column(1);
        trapezoid_slopes.set_column(1, &negative_slopes);

        let mut output_box = *input_domain.aabb();
        for _ in 0..steps {
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
                self.chunk_size,
            );
            std::mem::swap(input_domain, output_domain);
        }
        std::mem::swap(input_domain, output_domain);
    }
}
