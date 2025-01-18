use crate::domain::*;
use crate::fft_solver::*;
use crate::stencil::*;
use crate::util::*;

pub struct APSolver<
    'a,
    BC,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    pub solver_context:
        APSolverContext<'a, BC, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
}

impl<
        'a,
        BC,
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    > APSolver<'a, BC, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    pub fn new(
        bc: &'a BC,
        stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        aabb: AABB<GRID_DIMENSION>,
        steps: usize,
        plan_type: PlanType,
        cutoff: i32,
        ratio: f64,
        chunk_size: usize,
    ) -> Self {
        // Create our plan and convolution_store
        let planner = APPlanner::new(
            stencil, aabb, steps, plan_type, cutoff, ratio, chunk_size,
        );
        let planner_result = planner.finish();
        let plan = planner_result.plan;
        let convolution_store = planner_result.convolution_store;
        let stencil_slopes = planner_result.stencil_slopes;

        let (node_scratch_descriptors, scratch_space) =
            APScratchBuilder::build(&plan);

        let direct_frustrum_solver = DirectFrustrumSolver {
            bc,
            stencil,
            stencil_slopes,
            chunk_size,
        };

        let solver_context = APSolverContext {
            direct_frustrum_solver,
            convolution_store,
            plan,
            node_scratch_descriptors,
            scratch_space,
            chunk_size,
        };

        APSolver { solver_context }
    }

    pub fn apply(
        &self,
        input_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
    ) {
        self.solver_context.solve_root(input_domain, output_domain);
    }
}
