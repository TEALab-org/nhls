use crate::domain::*;
use crate::fft_solver::*;
use crate::stencil::*;
use crate::util::*;

struct APSolverContext<
    'a,
    const GRID_DIMENSION: usize,
    BC: BCCheck<GRID_DIMENSION>,
> {
    bc: &'a BC,
    stencil_slopes: &'a Bounds<GRID_DIMENSION>,
    convolution_store: &'a ConvolutionStore,
    plan: &'a APPlan<GRID_DIMENSION>,
    node_block_requirements: Vec<usize>,
    chunk_size: usize,
}

impl<'a, const GRID_DIMENSION: usize, BC: BCCheck<GRID_DIMENSION>>
    APSolverContext<'a, GRID_DIMENSION, BC>
{
    pub fn solve_root<DomainType: DomainView<GRID_DIMENSION>>(
        &self,
        input: &mut DomainType,
        output: &mut DomainType,
        scratch: &'a mut APNodeScratch<'a>,
    ) {
        //
    }

    pub fn solve_direct<'b>(
        &self,
        node_id: NodeId,
        input: &mut SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
        scratch: &'a mut APNodeScratch<'a>,
    ) {
        let direct_solve = self.plan.unwrap_direct_node(node_id);

        // Allocate io buffers
        let ([mut input_domain, mut output_domain], _remainder_) = scratch
            .split_io_domains(
                // copy input
                direct_solve.input_aabb,
            );
        input_domain.par_from_superset(input, self.chunk_size);

        // invoke direct solver

        // copy output to output
        output.par_set_subdomain(&output_domain, self.chunk_size);
    }
}

pub struct APSolver<
    'a,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
{
    pub stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
    pub aabb: AABB<GRID_DIMENSION>,
    pub steps: usize,
    pub convolution_store: ConvolutionStore,
    pub plan: APPlan<GRID_DIMENSION>,
    pub chunk_size: usize,
}

impl<
        'a,
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    > APSolver<'a, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
{
    pub fn new(
        stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        aabb: AABB<GRID_DIMENSION>,
        steps: usize,
        plan_type: PlanType,
        cutoff: i32,
        ratio: f64,
        chunk_size: usize,
    ) -> Self {
        let planner = APPlanner::new(
            stencil, aabb, steps, plan_type, cutoff, ratio, chunk_size,
        );
        let planner_result = planner.finish();

        APSolver {
            stencil,
            stencil_slopes: planner_result.stencil_slopes,
            aabb,
            steps,
            convolution_store: planner_result.convolution_store,
            plan: planner_result.plan,
            chunk_size,
        }
    }

    pub fn apply<DomainType: DomainView<GRID_DIMENSION>>(
        &mut self,
        _input: &mut DomainType,
        _output: &mut DomainType,
    ) {
    }
}
