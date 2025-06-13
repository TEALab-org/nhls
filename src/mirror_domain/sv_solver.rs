use crate::ap_solver::ap_periodic_ops::*;
use crate::ap_solver::ap_periodic_ops_builder::*;
use crate::ap_solver::generate_plan::*;
use crate::ap_solver::index_types::*;
use crate::ap_solver::plan::*;
use crate::ap_solver::planner::*;
use crate::ap_solver::scratch::*;
use crate::ap_solver::scratch_builder::*;
use crate::domain::*;
use crate::mirror_domain::*;
use crate::stencil::*;
use crate::util::*;

pub struct SVSolver<
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    SolverType: SVDirectSolver<GRID_DIMENSION> + Send + Sync,
> {
    pub direct_frustrum_solver: SolverType,
    pub convolution_store: ApPeriodicOps,
    pub remainder_convolution_store: ApPeriodicOps,
    pub plan: Plan<GRID_DIMENSION>,
    pub node_scratch_descriptors: Vec<ScratchDescriptor>,
    pub scratch_space_1: Scratch,
    pub scratch_space_2: Scratch,
    pub chunk_size: usize,
}

impl<
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        SolverType: SVDirectSolver<GRID_DIMENSION> + Send + Sync,
    > SVSolver<GRID_DIMENSION, NEIGHBORHOOD_SIZE, SolverType>
{
    pub fn new(
        stencil: &Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        params: &PlannerParameters<GRID_DIMENSION>,
        solver: SolverType,
    ) -> Self {
        let create_ops_builder = || ApPeriodicOpsBuilder::new(stencil, params);
        let planner_result = generate_plan(stencil, create_ops_builder, params);

        // Create our plan and convolution_store
        let plan = planner_result.plan;
        let complex_buffer_type = ComplexBufferType::DomainOnly;
        let (node_scratch_descriptors, scratch_space_1, scratch_space_2) =
            ScratchBuilder::build_double(&plan, complex_buffer_type);

        SVSolver {
            direct_frustrum_solver: solver,
            convolution_store: planner_result.periodic_ops,
            remainder_convolution_store: planner_result.remainder_periodic_ops,
            plan,
            node_scratch_descriptors,
            scratch_space_1,
            scratch_space_2,
            chunk_size: params.chunk_size,
        }
    }

    pub fn apply<'b>(
        &'b self,
        input_domain_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        input_domain_2: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain_2: &mut SliceDomain<'b, GRID_DIMENSION>,

        global_time: usize,
    ) {
        self.solve_root(
            input_domain_1,
            output_domain_1,
            input_domain_2,
            output_domain_2,
            global_time,
        );
    }

    fn get_input_output_1<'b>(
        &'b self,
        node_id: usize,
        aabb: &AABB<GRID_DIMENSION>,
    ) -> (
        SliceDomain<'b, GRID_DIMENSION>,
        SliceDomain<'b, GRID_DIMENSION>,
    ) {
        let scratch_descriptor = &self.node_scratch_descriptors[node_id];
        let input_buffer = self.scratch_space_1.unsafe_get_buffer(
            scratch_descriptor.input_offset,
            scratch_descriptor.real_buffer_size,
        );
        let output_buffer = self.scratch_space_1.unsafe_get_buffer(
            scratch_descriptor.output_offset,
            scratch_descriptor.real_buffer_size,
        );
        debug_assert!(input_buffer.len() >= aabb.buffer_size());
        debug_assert!(output_buffer.len() >= aabb.buffer_size());

        let input_domain = SliceDomain::new(*aabb, input_buffer);
        let output_domain = SliceDomain::new(*aabb, output_buffer);
        (input_domain, output_domain)
    }

    fn get_input_output_2<'b>(
        &'b self,
        node_id: usize,
        aabb: &AABB<GRID_DIMENSION>,
    ) -> (
        SliceDomain<'b, GRID_DIMENSION>,
        SliceDomain<'b, GRID_DIMENSION>,
    ) {
        let scratch_descriptor = &self.node_scratch_descriptors[node_id];
        let input_buffer = self.scratch_space_2.unsafe_get_buffer(
            scratch_descriptor.input_offset,
            scratch_descriptor.real_buffer_size,
        );
        let output_buffer = self.scratch_space_2.unsafe_get_buffer(
            scratch_descriptor.output_offset,
            scratch_descriptor.real_buffer_size,
        );
        debug_assert!(input_buffer.len() >= aabb.buffer_size());
        debug_assert!(output_buffer.len() >= aabb.buffer_size());

        let input_domain = SliceDomain::new(*aabb, input_buffer);
        let output_domain = SliceDomain::new(*aabb, output_buffer);
        (input_domain, output_domain)
    }

    fn get_complex_1(&self, node_id: usize) -> &mut [c64] {
        let scratch_descriptor = &self.node_scratch_descriptors[node_id];
        self.scratch_space_1.unsafe_get_buffer(
            scratch_descriptor.complex_offset,
            scratch_descriptor.complex_buffer_size,
        )
    }

    fn get_complex_2(&self, node_id: usize) -> &mut [c64] {
        let scratch_descriptor = &self.node_scratch_descriptors[node_id];
        self.scratch_space_2.unsafe_get_buffer(
            scratch_descriptor.complex_offset,
            scratch_descriptor.complex_buffer_size,
        )
    }

    pub fn solve_root<'b>(
        &'b self,
        input_domain_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        input_domain_2: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain_2: &mut SliceDomain<'b, GRID_DIMENSION>,
        mut global_time: usize,
    ) {
        let repeat_solve = self.plan.unwrap_repeat_node(self.plan.root);
        let repeat_periodic_solve =
            self.plan.unwrap_periodic_node(repeat_solve.node);
        let repeat_steps = repeat_periodic_solve.steps;

        for _ in 0..repeat_solve.n {
            self.periodic_solve_preallocated_io(
                repeat_solve.node,
                false,
                input_domain_1,
                output_domain_1,
                input_domain_2,
                output_domain_2,
                global_time,
            );
            global_time += repeat_steps;
            std::mem::swap(input_domain_1, output_domain_1);
            std::mem::swap(input_domain_2, output_domain_2);
        }
        if let Some(next) = repeat_solve.next {
            self.periodic_solve_preallocated_io(
                next,
                false,
                input_domain_1,
                output_domain_1,
                input_domain_2,
                output_domain_2,
                global_time,
            )
        } else {
            std::mem::swap(input_domain_1, output_domain_1);
            std::mem::swap(input_domain_2, output_domain_2);
        }
    }

    pub fn unknown_solve_allocate_io<'b>(
        &self,
        node_id: NodeId,
        input_1: &SliceDomain<'b, GRID_DIMENSION>,
        output_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        input_2: &SliceDomain<'b, GRID_DIMENSION>,
        output_2: &mut SliceDomain<'b, GRID_DIMENSION>,
        global_time: usize,
    ) {
        match self.plan.get_node(node_id) {
            PlanNode::DirectSolve(_) => {
                self.direct_solve_allocate_io(
                    node_id,
                    input_1,
                    output_1,
                    input_2,
                    output_2,
                    global_time,
                );
            }
            PlanNode::PeriodicSolve(_) => {
                self.periodic_solve_allocate_io(
                    node_id,
                    input_1,
                    output_1,
                    input_2,
                    output_2,
                    global_time,
                );
            }
            PlanNode::Repeat(_) => {
                panic!("ERROR: Not expecting repeat node");
            }
            PlanNode::Range(_) => {
                panic!("ERROR: Not expecting range node");
            }
        }
    }

    pub fn unknown_solve_preallocated_io<'b>(
        &self,
        node_id: NodeId,
        input_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        input_2: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_2: &mut SliceDomain<'b, GRID_DIMENSION>,
        global_time: usize,
    ) {
        match self.plan.get_node(node_id) {
            PlanNode::DirectSolve(_) => {
                self.direct_solve_preallocated_io(
                    node_id,
                    input_1,
                    output_1,
                    input_2,
                    output_2,
                    global_time,
                );
            }
            PlanNode::PeriodicSolve(_) => {
                self.periodic_solve_preallocated_io(
                    node_id,
                    true,
                    input_1,
                    output_1,
                    input_2,
                    output_2,
                    global_time,
                );
            }
            PlanNode::Repeat(_) => {
                panic!("ERROR: Not expecting repeat node");
            }
            PlanNode::Range(_) => {
                panic!("ERROR: Not expecting range node");
            }
        }
    }

    pub fn periodic_solve_preallocated_io<'b>(
        &self,
        node_id: NodeId,
        resize: bool,
        input_domain_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        input_domain_2: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain_2: &mut SliceDomain<'b, GRID_DIMENSION>,
        mut global_time: usize,
    ) {
        let periodic_solve = self.plan.unwrap_periodic_node(node_id);
        std::mem::swap(input_domain_1, output_domain_1);
        std::mem::swap(input_domain_2, output_domain_2);

        input_domain_1.set_aabb(periodic_solve.input_aabb);
        input_domain_1.par_from_superset(output_domain_1, self.chunk_size);
        output_domain_1.set_aabb(periodic_solve.input_aabb);

        input_domain_2.set_aabb(periodic_solve.input_aabb);
        input_domain_2.par_from_superset(output_domain_2, self.chunk_size);
        output_domain_2.set_aabb(periodic_solve.input_aabb);

        // Apply convolution
        {
            let convolution_op =
                self.convolution_store.get(periodic_solve.convolution_id);
            convolution_op.apply(
                input_domain_1,
                output_domain_2,
                self.get_complex_1(node_id),
                self.chunk_size,
            );
        }

        {
            let convolution_op =
                self.convolution_store.get(periodic_solve.convolution_id);
            convolution_op.apply(
                input_domain_1,
                output_domain_2,
                self.get_complex_2(node_id),
                self.chunk_size,
            );
        }

        // Boundary
        // In a rayon scope, we fork for each of the boundary solves,
        // each of which will fill in their part of of output_domain
        {
            let input_domain_const_1: &SliceDomain<'b, GRID_DIMENSION> =
                input_domain_1;
            let input_domain_const_2: &SliceDomain<'b, GRID_DIMENSION> =
                input_domain_2;

            rayon::scope(|s| {
                for node_id in periodic_solve.boundary_nodes.clone() {
                    // Our plan should provide the guarantee that
                    // that boundary nodes have mutually exclusive
                    // access to the output_domain
                    let mut node_output_1 = output_domain_1.unsafe_mut_access();
                    let mut node_output_2 = output_domain_2.unsafe_mut_access();

                    // Each boundary solve will need
                    // new input / output domains from the scratch space
                    s.spawn(move |_| {
                        self.unknown_solve_allocate_io(
                            node_id,
                            input_domain_const_1,
                            &mut node_output_1,
                            input_domain_const_2,
                            &mut node_output_2,
                            global_time,
                        );
                    });
                }
            });
        }

        if resize {
            std::mem::swap(input_domain_1, output_domain_1);
            output_domain_1.set_aabb(periodic_solve.output_aabb);
            output_domain_1.par_from_superset(input_domain_1, self.chunk_size);
            input_domain_1.set_aabb(periodic_solve.output_aabb);

            std::mem::swap(input_domain_2, output_domain_2);
            output_domain_2.set_aabb(periodic_solve.output_aabb);
            output_domain_2.par_from_superset(input_domain_2, self.chunk_size);
            input_domain_2.set_aabb(periodic_solve.output_aabb);
        }

        // call time cut if needed
        if let Some(next_id) = periodic_solve.time_cut {
            global_time += periodic_solve.steps;
            std::mem::swap(input_domain_1, output_domain_1);
            std::mem::swap(input_domain_2, output_domain_2);
            self.unknown_solve_preallocated_io(
                next_id,
                input_domain_1,
                output_domain_1,
                input_domain_2,
                output_domain_2,
                global_time,
            );
        }
    }

    pub fn periodic_solve_allocate_io<'b>(
        &self,
        node_id: NodeId,
        input_1: &SliceDomain<'b, GRID_DIMENSION>,
        output_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        input_2: &SliceDomain<'b, GRID_DIMENSION>,
        output_2: &mut SliceDomain<'b, GRID_DIMENSION>,
        global_time: usize,
    ) {
        let periodic_solve = self.plan.unwrap_periodic_node(node_id);

        let (mut input_domain_1, mut output_domain_1) =
            self.get_input_output_1(node_id, &periodic_solve.input_aabb);
        let (mut input_domain_2, mut output_domain_2) =
            self.get_input_output_2(node_id, &periodic_solve.input_aabb);

        // copy input
        input_domain_1.par_from_superset(input_1, self.chunk_size);
        input_domain_2.par_from_superset(input_2, self.chunk_size);

        self.periodic_solve_preallocated_io(
            node_id,
            true,
            &mut input_domain_1,
            &mut output_domain_1,
            &mut input_domain_2,
            &mut output_domain_2,
            global_time,
        );

        // copy output to output
        output_1.par_set_subdomain(&output_domain_1, self.chunk_size);
        output_2.par_set_subdomain(&output_domain_2, self.chunk_size);
    }

    pub fn direct_solve_allocate_io<'b>(
        &self,
        node_id: NodeId,
        input_1: &SliceDomain<'b, GRID_DIMENSION>,
        output_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        input_2: &SliceDomain<'b, GRID_DIMENSION>,
        output_2: &mut SliceDomain<'b, GRID_DIMENSION>,
        global_time: usize,
    ) {
        let direct_solve = self.plan.unwrap_direct_node(node_id);

        let (mut input_domain_1, mut output_domain_1) =
            self.get_input_output_1(node_id, &direct_solve.input_aabb);
        let (mut input_domain_2, mut output_domain_2) =
            self.get_input_output_2(node_id, &direct_solve.input_aabb);

        // copy input
        input_domain_1.par_from_superset(input_1, self.chunk_size);
        input_domain_2.par_from_superset(input_2, self.chunk_size);

        self.direct_solve_preallocated_io(
            node_id,
            &mut input_domain_1,
            &mut output_domain_1,
            &mut input_domain_2,
            &mut output_domain_2,
            global_time,
        );

        // copy output to output
        rayon::scope(|s| {
            s.spawn(move |_| {
                output_1
                    .par_set_from(&output_domain_1, &direct_solve.output_aabb);
            });
            s.spawn(move |_| {
                output_2
                    .par_set_from(&output_domain_2, &direct_solve.output_aabb);
            });
        });
    }

    pub fn direct_solve_preallocated_io<'b>(
        &self,
        node_id: NodeId,
        input_domain_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        input_domain_2: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain_2: &mut SliceDomain<'b, GRID_DIMENSION>,

        global_time: usize,
    ) {
        let direct_solve = self.plan.unwrap_direct_node(node_id);

        debug_assert!(input_domain_1
            .aabb()
            .contains_aabb(&direct_solve.input_aabb));
        debug_assert!(input_domain_2
            .aabb()
            .contains_aabb(&direct_solve.input_aabb));

        // For time-cuts, the provided domains
        // will not have the expected sizes.
        // All we know is that the provided input domain contains
        // the expected input domain
        std::mem::swap(input_domain_1, output_domain_1);
        input_domain_1.set_aabb(direct_solve.input_aabb);
        input_domain_1.par_from_superset(output_domain_1, self.chunk_size);
        output_domain_1.set_aabb(direct_solve.input_aabb);
        debug_assert_eq!(*input_domain_1.aabb(), direct_solve.input_aabb);

        std::mem::swap(input_domain_2, output_domain_2);
        input_domain_2.set_aabb(direct_solve.input_aabb);
        input_domain_2.par_from_superset(output_domain_2, self.chunk_size);
        output_domain_2.set_aabb(direct_solve.input_aabb);
        debug_assert_eq!(*input_domain_2.aabb(), direct_solve.input_aabb);

        // invoke direct solver
        self.direct_frustrum_solver.apply(
            input_domain_1,
            output_domain_1,
            input_domain_2,
            output_domain_2,
            &direct_solve.sloped_sides,
            direct_solve.steps,
            global_time,
            direct_solve.threads,
        );
    }
}
