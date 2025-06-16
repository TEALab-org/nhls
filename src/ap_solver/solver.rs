use crate::ap_solver::direct_solver::*;
use crate::ap_solver::index_types::*;
use crate::ap_solver::periodic_ops::*;
use crate::ap_solver::plan::*;
use crate::ap_solver::planner::*;
use crate::ap_solver::scratch::*;
use crate::ap_solver::scratch_builder::*;
use crate::ap_solver::solver_parameters::*;
use crate::SolverInterface;

use crate::domain::*;

use crate::mem_fmt::*;
use crate::util::*;
use std::io::prelude::*;

impl<
        const GRID_DIMENSION: usize,
        DirectSolverType: DirectSolver<GRID_DIMENSION>,
        PeriodicOpsType: PeriodicOps<GRID_DIMENSION>,
    > SolverInterface<GRID_DIMENSION>
    for Solver<GRID_DIMENSION, DirectSolverType, PeriodicOpsType>
{
    fn apply<'a>(
        &mut self,
        input_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        global_time: usize,
    ) {
        self.apply(input_domain, output_domain, global_time);
    }

    fn print_report(&self) {
        self.print_report();
    }

    fn to_dot_file<P: AsRef<std::path::Path>>(&self, path: &P) {
        self.plan.to_dot_file(path);
    }
}

pub struct Solver<
    const GRID_DIMENSION: usize,
    DirectSolverType: DirectSolver<GRID_DIMENSION>,
    PeriodicOpsType: PeriodicOps<GRID_DIMENSION>,
> {
    pub direct_solver: DirectSolverType,
    pub periodic_ops: PeriodicOpsType,
    pub remainder_periodic_ops: PeriodicOpsType,
    pub plan: Plan<GRID_DIMENSION>,
    pub node_scratch_descriptors: Vec<ScratchDescriptor>,
    pub scratch_space: Scratch,
    pub central_global_time: usize,
    pub chunk_size: usize,
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        DirectSolverType: DirectSolver<GRID_DIMENSION>,
        PeriodicOpsType: PeriodicOps<GRID_DIMENSION>,
    > Solver<GRID_DIMENSION, DirectSolverType, PeriodicOpsType>
{
    pub fn new(
        direct_solver: DirectSolverType,
        params: &SolverParameters<GRID_DIMENSION>,
        planner_result: PlannerResult<GRID_DIMENSION, PeriodicOpsType>,
        complex_buffer_type: ComplexBufferType,
    ) -> Self {
        profiling::scope!("ap_solver::new");

        let (node_scratch_descriptors, scratch_space) =
            ScratchBuilder::build(&planner_result.plan, complex_buffer_type);

        Solver {
            direct_solver,
            periodic_ops: planner_result.periodic_ops,
            remainder_periodic_ops: planner_result.remainder_periodic_ops,
            plan: planner_result.plan,
            node_scratch_descriptors,
            scratch_space,
            chunk_size: params.chunk_size,
            central_global_time: 0,
        }
    }

    pub fn print_report(&self) {
        println!("AP Solver Report:");
        println!("  - plan size: {}", self.plan.len());
        println!(
            "  - scratch size: {}",
            human_readable_bytes(self.scratch_space.size)
        );
    }

    pub fn apply(
        &mut self,
        input_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        global_time: usize,
    ) {
        profiling::scope!("ap_solver::apply");
        self.central_global_time = global_time;
        self.solve_root(input_domain, output_domain, global_time);
    }

    pub fn to_dot_file<P: AsRef<std::path::Path>>(&self, path: &P) {
        self.plan.to_dot_file(path);
    }

    pub fn scratch_descriptor_file<P: AsRef<std::path::Path>>(&self, path: &P) {
        let mut writer =
            std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        for (i, d) in self.node_scratch_descriptors.iter().enumerate() {
            writeln!(writer, "n_id: {} -- {:?}", i, d).unwrap();
        }
    }

    fn p_get_input_output<'b>(
        &'b self,
        node_id: usize,
        aabb: &AABB<GRID_DIMENSION>,
    ) -> (
        SliceDomain<'b, GRID_DIMENSION>,
        SliceDomain<'b, GRID_DIMENSION>,
    ) {
        let scratch_descriptor = &self.node_scratch_descriptors[node_id];
        let input_buffer = self.scratch_space.unsafe_get_buffer(
            scratch_descriptor.p_i,
            scratch_descriptor.real_buffer_size,
        );
        let output_buffer = self.scratch_space.unsafe_get_buffer(
            scratch_descriptor.p_o,
            scratch_descriptor.real_buffer_size,
        );

        debug_assert!(input_buffer.len() >= aabb.buffer_size());
        debug_assert!(output_buffer.len() >= aabb.buffer_size());

        let input_domain = SliceDomain::new(*aabb, input_buffer);
        let output_domain = SliceDomain::new(*aabb, output_buffer);
        (input_domain, output_domain)
    }

    fn get_input_output<'b>(
        &'b self,
        node_id: usize,
        aabb: &AABB<GRID_DIMENSION>,
    ) -> (
        SliceDomain<'b, GRID_DIMENSION>,
        SliceDomain<'b, GRID_DIMENSION>,
    ) {
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
        &mut self,
        input_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        mut global_time: usize,
    ) {
        let repeat_solve = self.plan.unwrap_repeat_node(self.plan.root);
        let repeat_periodic_solve =
            self.plan.unwrap_periodic_node(repeat_solve.node);
        let repeat_steps = repeat_periodic_solve.steps;

        for _ in 0..repeat_solve.n {
            self.periodic_ops.build_ops(global_time);
            self.periodic_solve(
                repeat_solve.node,
                input_domain,
                output_domain,
                global_time,
            );
            global_time += repeat_steps;
            std::mem::swap(input_domain, output_domain);
            profiling::finish_frame!();
        }
        if let Some(next) = repeat_solve.next {
            std::mem::swap(
                &mut self.periodic_ops,
                &mut self.remainder_periodic_ops,
            );
            self.remainder_periodic_ops.build_ops(global_time);
            self.periodic_solve(next, input_domain, output_domain, global_time);
            std::mem::swap(
                &mut self.periodic_ops,
                &mut self.remainder_periodic_ops,
            );
            profiling::finish_frame!();
        } else {
            std::mem::swap(input_domain, output_domain);
        }
    }

    pub fn unknown_solve_allocate_io(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'a, GRID_DIMENSION>,
        output: &mut SliceDomain<'a, GRID_DIMENSION>,
        global_time: usize,
    ) {
        match self.plan.get_node(node_id) {
            PlanNode::DirectSolve(_) => {
                self.direct_solve_allocate_io(
                    node_id,
                    input,
                    output,
                    global_time,
                );
            }
            PlanNode::PeriodicSolve(_) => {
                self.periodic_solve_allocate_io(
                    node_id,
                    input,
                    output,
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

    pub fn unknown_solve_preallocated_io(
        &self,
        node_id: NodeId,
        input: &mut SliceDomain<'a, GRID_DIMENSION>,
        output: &mut SliceDomain<'a, GRID_DIMENSION>,
        global_time: usize,
    ) {
        match self.plan.get_node(node_id) {
            PlanNode::DirectSolve(_) => {
                self.direct_solve_preallocated_io(
                    node_id,
                    input,
                    output,
                    global_time,
                );
            }
            PlanNode::PeriodicSolve(_) => {
                self.periodic_solve_preallocated_io(
                    node_id,
                    input,
                    output,
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

    pub fn periodic_solve_preallocated_io(
        &self,
        node_id: NodeId,
        input_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        mut global_time: usize,
    ) {
        profiling::scope!("ap_solver::periodic_solve_preallocated_io");
        let periodic_solve = self.plan.unwrap_periodic_node(node_id);
        std::mem::swap(input_domain, output_domain);
        input_domain.set_aabb(periodic_solve.input_aabb);
        input_domain.par_from_superset(output_domain, self.chunk_size);
        output_domain.set_aabb(periodic_solve.input_aabb);

        self.periodic_solve(node_id, input_domain, output_domain, global_time);

        std::mem::swap(input_domain, output_domain);
        output_domain.set_aabb(periodic_solve.output_aabb);
        output_domain.par_from_superset(input_domain, self.chunk_size);
        input_domain.set_aabb(periodic_solve.output_aabb);

        // call time cut if needed
        if let Some(next_id) = periodic_solve.time_cut {
            global_time += periodic_solve.steps;
            std::mem::swap(input_domain, output_domain);
            self.unknown_solve_preallocated_io(
                next_id,
                input_domain,
                output_domain,
                global_time,
            );
        }
    }

    pub fn periodic_solve_allocate_io(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'a, GRID_DIMENSION>,
        output: &mut SliceDomain<'a, GRID_DIMENSION>,
        mut global_time: usize,
    ) {
        profiling::scope!("ap_solver::periodic_solve_allocate_io");

        let periodic_solve = self.plan.unwrap_periodic_node(node_id);

        let (mut input_domain, mut output_domain) =
            self.get_input_output(node_id, &periodic_solve.input_aabb);

        // copy input
        input_domain.par_from_superset(input, self.chunk_size);

        self.periodic_solve(
            node_id,
            &mut input_domain,
            &mut output_domain,
            global_time,
        );

        // TODO: this could get merged with the copy instruction below
        // if we had an extra method for copying from AABB
        std::mem::swap(&mut input_domain, &mut output_domain);
        output_domain.set_aabb(periodic_solve.output_aabb);
        output_domain.par_from_superset(&input_domain, self.chunk_size);
        input_domain.set_aabb(periodic_solve.output_aabb);

        // call time cut if needed
        if let Some(next_id) = periodic_solve.time_cut {
            global_time += periodic_solve.steps;
            std::mem::swap(&mut input_domain, &mut output_domain);
            self.unknown_solve_preallocated_io(
                next_id,
                &mut input_domain,
                &mut output_domain,
                global_time,
            );
        }

        // copy output to output
        output.par_set_subdomain(&output_domain, self.chunk_size);
    }

    pub fn periodic_solve(
        &self,
        node_id: NodeId,
        input_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        global_time: usize,
    ) {
        profiling::scope!("ap_solver::periodic_solve");
        let periodic_solve = self.plan.unwrap_periodic_node(node_id);

        // Boundary
        // In a rayon scope, we fork for each of the boundary solves,
        // each of which will fill in their part of of output_domain
        {
            let input_domain_const: &SliceDomain<'a, GRID_DIMENSION> =
                input_domain;
            let output_domain_const: &SliceDomain<'a, GRID_DIMENSION> =
                output_domain;

            rayon::scope(|s| {
                s.spawn(move |_| {
                    // Our plan should provide the guarantee that
                    // that boundary nodes have mutually exclusive
                    // access to the output_domain
                    let mut node_output =
                        output_domain_const.unsafe_mut_access();

                    let (mut p_input_domain, mut p_output_domain) = self
                        .p_get_input_output(
                            node_id,
                            &periodic_solve.input_aabb,
                        );

                    // copy input
                    p_input_domain
                        .par_from_superset(input_domain_const, self.chunk_size);

                    // Apply convolution
                    self.periodic_ops.apply_operation(
                        periodic_solve.convolution_id,
                        &mut p_input_domain,
                        &mut p_output_domain,
                        self.get_complex(node_id),
                        self.central_global_time,
                        self.chunk_size,
                    );

                    p_input_domain.set_aabb(periodic_solve.output_aabb);
                    p_input_domain
                        .par_from_superset(&p_output_domain, self.chunk_size);
                    node_output
                        .par_set_subdomain(&p_input_domain, self.chunk_size);
                });

                for node_id in periodic_solve.boundary_nodes.clone() {
                    // Our plan should provide the guarantee that
                    // that boundary nodes have mutually exclusive
                    // access to the output_domain
                    let mut node_output =
                        output_domain_const.unsafe_mut_access();

                    // Each boundary solve will need
                    // new input / output domains from the scratch space
                    s.spawn(move |_| {
                        self.unknown_solve_allocate_io(
                            node_id,
                            input_domain_const,
                            &mut node_output,
                            global_time,
                        );
                    });
                }
            });
        }
    }

    pub fn direct_solve_allocate_io<'b>(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
        global_time: usize,
    ) {
        profiling::scope!("ap_solver::direct_solve_allocate_io");

        let direct_solve = self.plan.unwrap_direct_node(node_id);

        let (mut input_domain, mut output_domain) =
            self.get_input_output(node_id, &direct_solve.input_aabb);

        // copy input
        input_domain.par_from_superset(input, self.chunk_size);

        // invoke direct solver
        self.direct_solver.apply(
            &mut input_domain,
            &mut output_domain,
            &direct_solve.sloped_sides,
            direct_solve.steps,
            global_time,
            direct_solve.threads,
        );

        // NOTE (rb): until opt solvers handle_frustrum solves...
        input_domain.set_aabb(direct_solve.output_aabb);
        input_domain.par_from_superset(&output_domain, self.chunk_size);
        std::mem::swap(&mut input_domain, &mut output_domain);

        debug_assert_eq!(
            direct_solve.output_aabb,
            *output_domain.aabb(),
            "ERROR: n_id: {}, Unexpected solve output",
            node_id
        );

        // copy output to output
        output.par_set_subdomain(&output_domain, self.chunk_size);
    }

    pub fn direct_solve_preallocated_io(
        &self,
        node_id: NodeId,
        input_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        global_time: usize,
    ) {
        profiling::scope!("ap_solver::direct_solve_preallocated_io");
        let direct_solve = self.plan.unwrap_direct_node(node_id);

        debug_assert!(input_domain
            .aabb()
            .contains_aabb(&direct_solve.input_aabb));

        // For time-cuts, the provided domains
        // will not have the expected sizes.
        // All we know is that the provided input domain contains
        // the expected input domain
        std::mem::swap(input_domain, output_domain);
        input_domain.set_aabb(direct_solve.input_aabb);
        input_domain.par_from_superset(output_domain, self.chunk_size);
        output_domain.set_aabb(direct_solve.input_aabb);
        debug_assert_eq!(*input_domain.aabb(), direct_solve.input_aabb);

        // invoke direct solver
        self.direct_solver.apply(
            input_domain,
            output_domain,
            &direct_solve.sloped_sides,
            direct_solve.steps,
            global_time,
            direct_solve.threads,
        );

        // NOTE (rb): until opt solvers handle_frustrum solves...
        input_domain.set_aabb(direct_solve.output_aabb);
        input_domain.par_from_superset(output_domain, self.chunk_size);
        std::mem::swap(input_domain, output_domain);

        debug_assert_eq!(
            direct_solve.output_aabb,
            *output_domain.aabb(),
            "ERROR: n_id: {}, Unexpected solve output",
            node_id
        );
    }
}
