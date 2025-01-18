use crate::domain::*;
use crate::fft_solver::*;
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
    node_scratch_descriptors: Vec<ScratchDescriptor>,
    scratch_space: ScratchSpace,
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
    fn get_input_output(
        &self,
        node_id: usize,
        aabb: &AABB<GRID_DIMENSION>,
    ) -> (SliceDomain<GRID_DIMENSION>, SliceDomain<GRID_DIMENSION>) {
        let scratch_descriptor = &self.node_scratch_descriptors[node_id];
        let input_buffer = self.scratch_space.unsafe_get_buffer(
            scratch_descriptor.input_offset,
            scratch_descriptor.real_buffer_size,
        );
        let output_buffer = self.scratch_space.unsafe_get_buffer(
            scratch_descriptor.output_offset,
            scratch_descriptor.real_buffer_size,
        );
        debug_assert!(input_buffer.len() >= aabb.buffer_size());
        debug_assert!(output_buffer.len() >= aabb.buffer_size());

        let input_domain = SliceDomain::new(*aabb, input_buffer);
        let output_domain = SliceDomain::new(*aabb, output_buffer);
        (input_domain, output_domain)
    }

    fn get_complex(&self, node_id: usize) -> &mut [c64] {
        let scratch_descriptor = &self.node_scratch_descriptors[node_id];
        self.scratch_space.unsafe_get_buffer(
            scratch_descriptor.complex_offset,
            scratch_descriptor.complex_buffer_size,
        )
    }

    pub fn solve_root(
        &self,
        input_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
    ) {
        let repeat_solve = self.plan.unwrap_repeat_node(self.plan.root);
        for _ in 0..repeat_solve.n {
            self.periodic_solve_preallocated_io(
                repeat_solve.node,
                input_domain,
                output_domain,
            );
            std::mem::swap(input_domain, output_domain);
        }
        if let Some(next) = repeat_solve.next {
            self.periodic_solve_preallocated_io(
                next,
                input_domain,
                output_domain,
            )
        } else {
            std::mem::swap(input_domain, output_domain);
        }
    }

    pub fn unknown_solve_allocate_io<'b>(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
    ) {
        match self.plan.get_node(node_id) {
            PlanNode::DirectSolve(_) => {
                self.direct_solve(node_id, input, output);
            }
            PlanNode::PeriodicSolve(_) => {
                self.periodic_solve_allocate_io(node_id, input, output);
            }
            PlanNode::Repeat(_) => {
                panic!("ERROR: Not expecting repeat node");
            }
        }
    }

    pub fn periodic_solve_preallocated_io<'b>(
        &self,
        node_id: NodeId,
        input_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
    ) {
        let periodic_solve = self.plan.unwrap_periodic_node(node_id);
        debug_assert_eq!(periodic_solve.input_aabb, *input_domain.aabb());
        debug_assert_eq!(periodic_solve.input_aabb, *output_domain.aabb());

        // Apply convolution
        {
            let convolution_op =
                self.convolution_store.get(periodic_solve.convolution_id);
            convolution_op.apply(
                input_domain,
                output_domain,
                self.get_complex(node_id),
                self.chunk_size,
            );
        }

        // Boundary
        // For each boundary node we need
        // block requirement
        // split that from scratch
        // get mutable output domain
        {
            let input_domain_const: &SliceDomain<'b, GRID_DIMENSION> =
                input_domain;
            rayon::scope(|s| {
                for node_id in periodic_solve.boundary_nodes.clone() {
                    let mut node_output = output_domain.unsafe_mut_access();
                    s.spawn(move |_| {
                        self.unknown_solve_allocate_io(
                            node_id,
                            input_domain_const,
                            &mut node_output,
                        );
                    });
                }
            });
        }

        // call time cut if needed
        if let Some(next_id) = periodic_solve.time_cut {
            std::mem::swap(input_domain, output_domain);
            self.periodic_solve_preallocated_io(
                next_id,
                input_domain,
                output_domain,
            );
        }
    }

    pub fn periodic_solve_allocate_io<'b>(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
    ) {
        let periodic_solve = self.plan.unwrap_periodic_node(node_id);
        let (mut input_domain, mut output_domain) =
            self.get_input_output(node_id, &periodic_solve.input_aabb);

        // copy input
        input_domain.par_from_superset(input, self.chunk_size);

        self.periodic_solve_preallocated_io(
            node_id,
            &mut input_domain,
            &mut output_domain,
        );
        // copy output to output
        output.par_set_subdomain(&output_domain, self.chunk_size);
    }

    pub fn direct_solve<'b>(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
    ) {
        let direct_solve = self.plan.unwrap_direct_node(node_id);

        let (mut input_domain, mut output_domain) =
            self.get_input_output(node_id, &direct_solve.input_aabb);

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
