use crate::domain::*;
use crate::fft_solver::*;
use crate::mem_fmt::*;
use crate::stencil::*;
use crate::util::*;
use std::io::prelude::*;

pub struct APSolver<
    'a,
    BC: BCCheck<GRID_DIMENSION>,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    BC: BCCheck<GRID_DIMENSION>,
{
    pub direct_frustrum_solver:
        DirectFrustrumSolver<'a, BC, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    pub convolution_store: ConvolutionStore,
    pub plan: APPlan<GRID_DIMENSION>,
    pub node_scratch_descriptors: Vec<ScratchDescriptor>,
    pub scratch_space: APScratch,
    pub chunk_size: usize,
}

impl<'a, BC, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>
    APSolver<'a, BC, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    BC: BCCheck<GRID_DIMENSION>,
{
    pub fn new(
        bc: &'a BC,
        stencil: &'a Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        aabb: AABB<GRID_DIMENSION>,
        steps: usize,
        params: &PlannerParameters,
    ) -> Self {
        // Create our plan and convolution_store
        let planner_result = create_ap_plan(stencil, aabb, steps, params);
        let plan = planner_result.plan;
        let convolution_store = planner_result.convolution_store;
        let stencil_slopes = planner_result.stencil_slopes;

        let (node_scratch_descriptors, scratch_space) =
            APScratchBuilder::build(&plan);

        let direct_frustrum_solver = DirectFrustrumSolver {
            bc,
            stencil,
            stencil_slopes,
            chunk_size: params.chunk_size,
        };

        APSolver {
            direct_frustrum_solver,
            convolution_store,
            plan,
            node_scratch_descriptors,
            scratch_space,
            chunk_size: params.chunk_size,
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
        &self,
        input_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        global_time: usize,
    ) {
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
                input_domain,
                output_domain,
                global_time,
            );
            global_time += repeat_steps;
            std::mem::swap(input_domain, output_domain);
        }
        if let Some(next) = repeat_solve.next {
            self.periodic_solve_preallocated_io(
                next,
                false,
                input_domain,
                output_domain,
                global_time,
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

    pub fn unknown_solve_preallocated_io<'b>(
        &self,
        node_id: NodeId,
        input: &mut SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
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
                    true,
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

    pub fn periodic_solve_preallocated_io<'b>(
        &self,
        node_id: NodeId,
        resize: bool,
        input_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        mut global_time: usize,
    ) {
        let periodic_solve = self.plan.unwrap_periodic_node(node_id);
        std::mem::swap(input_domain, output_domain);
        input_domain.set_aabb(periodic_solve.input_aabb);
        input_domain.par_from_superset(output_domain, self.chunk_size);
        output_domain.set_aabb(periodic_solve.input_aabb);

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
        // In a rayon scope, we fork for each of the boundary solves,
        // each of which will fill in their part of of output_domain
        {
            let input_domain_const: &SliceDomain<'b, GRID_DIMENSION> =
                input_domain;
            rayon::scope(|s| {
                for node_id in periodic_solve.boundary_nodes.clone() {
                    // Our plan should provide the guarantee that
                    // that boundary nodes have mutually exclusive
                    // access to the output_domain
                    let mut node_output = output_domain.unsafe_mut_access();

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

        if resize {
            std::mem::swap(input_domain, output_domain);
            output_domain.set_aabb(periodic_solve.output_aabb);
            output_domain.par_from_superset(input_domain, self.chunk_size);
            input_domain.set_aabb(periodic_solve.output_aabb);
        }

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

    pub fn periodic_solve_allocate_io<'b>(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
        global_time: usize,
    ) {
        let periodic_solve = self.plan.unwrap_periodic_node(node_id);

        let (mut input_domain, mut output_domain) =
            self.get_input_output(node_id, &periodic_solve.input_aabb);

        // copy input
        input_domain.par_from_superset(input, self.chunk_size);

        self.periodic_solve_preallocated_io(
            node_id,
            true,
            &mut input_domain,
            &mut output_domain,
            global_time,
        );

        // copy output to output
        output.par_set_subdomain(&output_domain, self.chunk_size);
    }

    pub fn direct_solve_allocate_io<'b>(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
        global_time: usize,
    ) {
        let direct_solve = self.plan.unwrap_direct_node(node_id);

        let (mut input_domain, mut output_domain) =
            self.get_input_output(node_id, &direct_solve.input_aabb);

        // copy input
        input_domain.par_from_superset(input, self.chunk_size);

        self.direct_solve_preallocated_io(
            node_id,
            &mut input_domain,
            &mut output_domain,
            global_time,
        );
        debug_assert_eq!(*output_domain.aabb(), direct_solve.output_aabb);

        // copy output to output
        output.par_set_subdomain(&output_domain, self.chunk_size);
    }

    pub fn direct_solve_preallocated_io<'b>(
        &self,
        node_id: NodeId,
        input_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        global_time: usize,
    ) {
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
        self.direct_frustrum_solver.apply(
            input_domain,
            output_domain,
            &direct_solve.sloped_sides,
            direct_solve.steps,
            global_time,
        );

        debug_assert_eq!(
            direct_solve.output_aabb,
            *output_domain.aabb(),
            "ERROR: n_id: {}, Unexpected solve output",
            node_id
        );
    }
}
