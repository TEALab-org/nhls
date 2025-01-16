use crate::domain::*;
use crate::fft_solver::*;
use crate::solver::trapezoid::*;
use crate::stencil::*;
use crate::util::*;

pub struct APSolverContext<
    'a,
    BC: BCCheck<GRID_DIMENSION>,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    direct_frustrum_solver: DirectFrustrumSolver<
        'a,
        BC,
        Operation,
        GRID_DIMENSION,
        NEIGHBORHOOD_SIZE,
    >,
    //stencil_slopes: &'a Bounds<GRID_DIMENSION>,
    convolution_store: &'a ConvolutionStore,
    plan: &'a APPlan<GRID_DIMENSION>,
    node_block_requirements: Vec<usize>,
    chunk_size: usize,
}

impl<
        'a,
        BC,
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    > APSolverContext<'a, BC, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    pub fn solve_root<DomainType: DomainView<GRID_DIMENSION>>(
        &self,
        input: &mut DomainType,
        output: &mut DomainType,
        scratch: &'a mut APNodeScratch<'a>,
    ) {
        //
    }

    pub fn periodic_solve_preallocated_io<'b>(
        &self,
        node_id: NodeId,
        input: &mut SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
        scratch: APNodeScratch<'a>,
    ) {
        let periodic_solve = self.plan.unwrap_periodic_node(node_id);
        debug_assert_eq!(periodic_solve.input_aabb, *input.aabb());
        debug_assert_eq!(periodic_solve.input_aabb, *output.aabb());
    }

    pub fn unknown_solve_allocate_io<'b>(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
        scratch: APNodeScratch<'a>,
    ) {
        match self.plan.get_node(node_id) {
            PlanNode::DirectSolve(_) => {
                self.direct_solve(node_id, input, output, scratch);
            }
            PlanNode::PeriodicSolve(_) => {
                self.periodic_solve_allocate_io(
                    node_id, input, output, scratch,
                );
            }
            PlanNode::Repeat(_) => {
                panic!("ERROR: Not expecting repeat node");
            }
        }
    }

    pub fn periodic_solve_allocate_io<'b>(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
        scratch: APNodeScratch<'a>,
    ) {
        let periodic_solve = self.plan.unwrap_periodic_node(node_id);

        // Allocate io buffers
        let ([mut input_domain, mut output_domain], mut scratch_remainder) =
            scratch.split_io_domains(periodic_solve.input_aabb);

        // copy input
        input_domain.par_from_superset(input, self.chunk_size);

        // Apply convolution
        {
            let convolution_op =
                self.convolution_store.get(periodic_solve.convolution_id);
            convolution_op.apply(
                &mut input_domain,
                &mut output_domain,
                scratch_remainder
                    .unsafe_complex_buffer(periodic_solve.input_aabb),
                self.chunk_size,
            );
        }

        // Boundary
        // For each boundary node we need
        // block requirement
        // split that from scratch
        // get mutable output domain
        rayon::scope(|s| {
            for node_id in periodic_solve.boundary_nodes.clone() {
                let block_requirement = self.node_block_requirements[node_id];
                let node_scratch: APNodeScratch<'_>;
                (node_scratch, scratch_remainder) =
                    scratch_remainder.split_scratch(block_requirement);
                let mut output_domain = output.unsafe_mut_access();
                s.spawn(move |_| {
                    self.unknown_solve_allocate_io(
                        node_id,
                        input,
                        &mut output_domain,
                        node_scratch,
                    );
                });
            }
        });

        // call time cut if needed

        // copy output to output
        output.par_set_subdomain(&output_domain, self.chunk_size);
    }

    pub fn direct_solve<'b>(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
        scratch: APNodeScratch<'a>,
    ) {
        let direct_solve = self.plan.unwrap_direct_node(node_id);

        // Allocate io buffers
        let ([mut input_domain, mut output_domain], _remainder_) =
            scratch.split_io_domains(direct_solve.input_aabb);

        // copy input
        input_domain.par_from_superset(input, self.chunk_size);

        // invoke direct solver
        self.direct_frustrum_solver.apply(
            &mut input_domain,
            &mut output_domain,
            &direct_solve.sloped_sides,
            direct_solve.steps,
        );
        debug_assert_eq!(direct_solve.output_aabb, *output_domain.aabb());

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
