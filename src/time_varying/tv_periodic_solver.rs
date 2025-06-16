use crate::ap_solver::scratch::*;
use crate::ap_solver::*;
use crate::domain::*;
use crate::par_slice;
use crate::solver_interface::*;
use crate::stencil::*;
use crate::time_varying::tv_periodic_solver_builder::*;
use crate::time_varying::*;
use crate::util::*;
use fftw::plan::*;
use rayon::prelude::*;

pub struct Base1Node {
    pub t: usize,
}

pub struct Base2Node<'a, const GRID_DIMENSION: usize> {
    pub t1: usize,
    pub t2: usize,
    pub s1: CircStencil<'a, GRID_DIMENSION>,
    pub s2: CircStencil<'a, GRID_DIMENSION>,
    pub c1: &'a mut [c64],
    pub c2: &'a mut [c64],
    pub plan_id: FFTPairId,
}

pub struct ConvolveNode<'a, const GRID_DIMENSION: usize> {
    pub n1: usize,
    pub n2: usize,
    pub s1: CircStencil<'a, GRID_DIMENSION>,
    pub s2: CircStencil<'a, GRID_DIMENSION>,
    pub c1: &'a mut [c64],
    pub c2: &'a mut [c64],
    pub plan_id: FFTPairId,
}

pub enum IntermediateNode<'a, const GRID_DIMENSION: usize> {
    Base1(Base1Node),
    Base2(Base2Node<'a, GRID_DIMENSION>),
    Convolve(ConvolveNode<'a, GRID_DIMENSION>),
}

impl<'a, const GRID_DIMENSION: usize> IntermediateNode<'a, GRID_DIMENSION> {
    pub fn clear_stencils(&mut self) {
        match self {
            IntermediateNode::Base1(_) => {}
            IntermediateNode::Base2(n) => {
                n.s1.clear();
                n.s2.clear();
            }
            IntermediateNode::Convolve(n) => {
                n.s1.clear();
                n.s2.clear();
            }
        }
    }
}

pub fn solve_base2_node<
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
>(
    node: &mut Base2Node<GRID_DIMENSION>,
    stencil: &StencilType,
    global_time: usize,
    fft_store: &FFTStore,
    chunk_size: usize,
) {
    node.s1.add_tv_stencil(stencil, global_time + node.t1);
    node.s2.add_tv_stencil(stencil, global_time + node.t2);

    // fft both
    let fft_pair = fft_store.get(node.plan_id);
    fft_pair
        .forward_plan
        .r2c(node.s1.domain.buffer_mut(), &mut node.c1)
        .unwrap();
    fft_pair
        .forward_plan
        .r2c(node.s2.domain.buffer_mut(), &mut node.c2)
        .unwrap();

    // Multiply in freq, return result to s1
    par_slice::multiply_by(&mut node.c1, &node.c2, chunk_size);
    fft_pair
        .backward_plan
        .c2r(&mut node.c1, node.s1.domain.buffer_mut())
        .unwrap();
    let n_r = node.s1.domain.aabb().buffer_size();
    par_slice::div(node.s1.domain.buffer_mut(), n_r as f64, chunk_size);
}

/// Helper function for solve_convolve_node
pub fn add_node_to_circ_stencil<
    'b,
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
>(
    s: &'b mut CircStencil<'a, GRID_DIMENSION>,
    id: usize,
    layer: &'b [IntermediateNode<'a, GRID_DIMENSION>],
    stencil: &StencilType,
    global_time: usize,
) {
    match &layer[id] {
        IntermediateNode::Base1(n) => {
            s.add_tv_stencil(stencil, global_time + n.t);
        }
        IntermediateNode::Base2(n) => {
            s.add_circ_stencil(&n.s1);
        }
        IntermediateNode::Convolve(n) => {
            s.add_circ_stencil(&n.s1);
        }
    }
}

pub fn solve_convolve_node<
    'solver_life,
    'node_borrow,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
>(
    node: &'node_borrow mut ConvolveNode<'solver_life, GRID_DIMENSION>,
    prev_layer: &'node_borrow [IntermediateNode<
        'solver_life,
        GRID_DIMENSION,
    >],
    stencil: &'node_borrow StencilType,
    global_time: usize,
    fft_store: &FFTStore,
    chunk_size: usize,
) {
    add_node_to_circ_stencil(
        &mut node.s1,
        node.n1,
        prev_layer,
        stencil,
        global_time,
    );
    add_node_to_circ_stencil(
        &mut node.s2,
        node.n2,
        prev_layer,
        stencil,
        global_time,
    );

    // fft both
    let fft_pair = fft_store.get(node.plan_id);
    fft_pair
        .forward_plan
        .r2c(node.s1.domain.buffer_mut(), node.c1)
        .unwrap();
    fft_pair
        .forward_plan
        .r2c(node.s2.domain.buffer_mut(), node.c2)
        .unwrap();

    // Multiply in freq, return result to s1
    par_slice::multiply_by(node.c1, node.c2, chunk_size);
    fft_pair
        .backward_plan
        .c2r(node.c1, node.s1.domain.buffer_mut())
        .unwrap();
    let n_r = node.s1.domain.aabb().buffer_size();
    par_slice::div(node.s1.domain.buffer_mut(), n_r as f64, chunk_size);
}

pub fn solve_base_layer<
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
>(
    stencil: &StencilType,
    threads: usize,
    global_time: usize,
    layer_nodes: &mut [IntermediateNode<GRID_DIMENSION>],
    fft_store: &FFTStore,
) {
    // TODO clear
    let chunk_size = 1.max(layer_nodes.len() / (threads * 2));
    layer_nodes
        .par_chunks_mut(chunk_size)
        .for_each(|layer_node_chunk| {
            for node in layer_node_chunk.iter_mut() {
                match node {
                    IntermediateNode::Base1(_) => {}
                    IntermediateNode::Base2(n) => {
                        solve_base2_node(
                            n,
                            stencil,
                            global_time,
                            fft_store,
                            10000,
                        );
                    }
                    IntermediateNode::Convolve(_) => {
                        panic!(
                            "ERROR: shouldn't be convolve nodes in base layer"
                        );
                    }
                }
            }
        });
}

pub fn solve_middle_layer<
    'solver_life,
    'node_borrow,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
>(
    stencil: &'node_borrow StencilType,
    threads: usize,
    global_time: usize,
    layer_nodes: &'node_borrow mut [IntermediateNode<
        'solver_life,
        GRID_DIMENSION,
    >],
    prev_layer_nodes: &'node_borrow [IntermediateNode<
        'solver_life,
        GRID_DIMENSION,
    >],
    fft_store: &FFTStore,
) {
    // TODO clear
    let chunk_size = 1.max(layer_nodes.len() / (threads * 2));
    layer_nodes
        .par_chunks_mut(chunk_size)
        .for_each(|layer_node_chunk| {
            for node in layer_node_chunk {
                match node {
                    IntermediateNode::Base1(_) => {}
                    IntermediateNode::Base2(n) => {
                        solve_base2_node(
                            n,
                            stencil,
                            global_time,
                            fft_store,
                            chunk_size,
                        );
                    }
                    IntermediateNode::Convolve(n) => {
                        solve_convolve_node(
                            n,
                            prev_layer_nodes,
                            stencil,
                            global_time,
                            fft_store,
                            chunk_size,
                        );
                    }
                }
            }
        });
}

pub struct TVPeriodicSolver<
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> {
    pub scratch: Scratch,
    pub stencil: &'a StencilType,
    pub c1: &'a mut [c64],
    pub c2: &'a mut [c64],
    pub fft_plans: FFTStore,
    pub intermediate_nodes: Vec<Vec<IntermediateNode<'a, GRID_DIMENSION>>>,
    pub chunk_size: usize,
    pub threads: usize,
    pub aabb: AABB<GRID_DIMENSION>,
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    > TVPeriodicSolver<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
{
    pub fn new(
        stencil: &'a StencilType,
        steps: usize,
        plan_type: PlanType,
        aabb: AABB<GRID_DIMENSION>,
        threads: usize,
    ) -> Self {
        let builder = TVPeriodicSolveBuilder::new(stencil, aabb);
        builder.build_solver(steps, threads, plan_type)
    }

    pub fn apply<'b, DomainType: DomainView<GRID_DIMENSION>>(
        &'b mut self,
        input: &mut DomainType,
        output: &mut DomainType,
        global_time: usize,
    ) {
        debug_assert_eq!(*input.aabb(), self.aabb);
        debug_assert_eq!(*output.aabb(), self.aabb);
        println!("Solver: Build base layer");
        // Build intermediate tree layers
        let base_layer_id = self.intermediate_nodes.len() - 1;
        solve_base_layer(
            self.stencil,
            self.threads,
            global_time,
            &mut self.intermediate_nodes[base_layer_id],
            &self.fft_plans,
        );
        for layer_id in (0..base_layer_id).rev() {
            println!("Solver: build layer: {}", layer_id);
            let (new, old) = self.intermediate_nodes.split_at_mut(layer_id + 1);
            solve_middle_layer(
                self.stencil,
                self.threads,
                global_time,
                new.last_mut().unwrap(),
                &old[0],
                &self.fft_plans,
            );
        }

        println!("Solver: build convolution");
        // Convolve all final nodes
        par_slice::set_value(
            self.c1,
            c64 { re: 1.0, im: 1.0 },
            self.chunk_size,
        );
        par_slice::set_value(output.buffer_mut(), 0.0, self.chunk_size);
        for node in &self.intermediate_nodes[0] {
            match node {
                IntermediateNode::Base1(n) => {
                    // Add stencil offsets to output
                    let weights = self.stencil.weights(global_time + n.t);
                    for i in 0..NEIGHBORHOOD_SIZE {
                        let offset = self.stencil.offsets()[i];
                        let weight = weights[i];
                        let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                        let periodic_coord = self.aabb.periodic_coord(&rn_i);
                        output.set_coord(&periodic_coord, weight);
                    }

                    // Run forward pass
                    self.fft_plans
                        .get(0)
                        .forward_plan
                        .r2c(output.buffer_mut(), self.c2)
                        .unwrap();
                    par_slice::multiply_by(self.c1, self.c2, self.chunk_size);

                    // Remove stencil offsets
                    for i in 0..NEIGHBORHOOD_SIZE {
                        let offset = self.stencil.offsets()[i];
                        let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                        let periodic_coord = self.aabb.periodic_coord(&rn_i);
                        output.set_coord(&periodic_coord, 0.0);
                    }
                }
                IntermediateNode::Base2(n) => {
                    // Add stencil offsets to output
                    for (offset, weight) in n.s1.to_offset_weights() {
                        let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                        let periodic_coord = self.aabb.periodic_coord(&rn_i);
                        output.set_coord(&periodic_coord, weight);
                    }

                    // Run forward pass
                    self.fft_plans
                        .get(0)
                        .forward_plan
                        .r2c(output.buffer_mut(), self.c2)
                        .unwrap();
                    par_slice::multiply_by(self.c1, self.c2, self.chunk_size);

                    // Remove stencil offsets
                    for (offset, _) in n.s1.to_offset_weights() {
                        let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                        let periodic_coord = self.aabb.periodic_coord(&rn_i);
                        output.set_coord(&periodic_coord, 0.0);
                    }
                }
                IntermediateNode::Convolve(n) => {
                    // Add stencil offsets to output
                    for (offset, weight) in n.s1.to_offset_weights() {
                        let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                        let periodic_coord = self.aabb.periodic_coord(&rn_i);
                        output.set_coord(&periodic_coord, weight);
                    }

                    // Run forward pass
                    self.fft_plans
                        .get(0)
                        .forward_plan
                        .r2c(output.buffer_mut(), self.c2)
                        .unwrap();
                    par_slice::multiply_by(self.c1, self.c2, self.chunk_size);

                    // Remove stencil offsets
                    for (offset, _) in n.s1.to_offset_weights() {
                        let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                        let periodic_coord = self.aabb.periodic_coord(&rn_i);
                        output.set_coord(&periodic_coord, 0.0);
                    }
                }
            }

            par_slice::multiply_by(self.c1, self.c2, self.chunk_size);
        }

        println!("Solver: apply convolution");

        // Apply global convolution
        self.fft_plans
            .get(0)
            .forward_plan
            .r2c(input.buffer_mut(), self.c1)
            .unwrap();

        // mul
        par_slice::multiply_by(self.c1, self.c2, self.chunk_size);

        // backward pass output
        self.fft_plans
            .get(0)
            .backward_plan
            .c2r(self.c1, output.buffer_mut())
            .unwrap();

        let n_r = output.aabb().buffer_size();
        par_slice::div(output.buffer_mut(), n_r as f64, self.chunk_size);

        for layer in self.intermediate_nodes.iter_mut() {
            layer.par_iter_mut().for_each(|n| n.clear_stencils());
        }
    }
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    > SolverInterface<GRID_DIMENSION>
    for TVPeriodicSolver<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
{
    fn apply<'b>(
        &mut self,
        input_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        global_time: usize,
    ) {
        self.apply(input_domain, output_domain, global_time);
    }

    fn print_report(&self) {
        println!("PeriodicSolver: No Report");
    }

    fn to_dot_file<P: AsRef<std::path::Path>>(&self, _path: &P) {
        eprintln!("WARNING: PerodicSolver cannot save to dot file");
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::mem_fmt::human_readable_bytes;

    use super::*;

    fn test_tree_size<const GRID_DIMENSION: usize>(
        stencil_slopes: Bounds<GRID_DIMENSION>,
        steps: usize,
    ) {
        let mut current_nodes = steps;
        let mut current_slopes = stencil_slopes;
        let mut layer = 1;
        while current_nodes != 1 {
            let new_layer_size = current_nodes / 2;
            let extra_node = current_nodes % 2 == 1;
            current_slopes += current_slopes;
            let aabb = slopes_to_circ_aabb(&current_slopes);
            // 2 input domains, one complex domain
            let node_size = 2 * aabb.buffer_size() * size_of::<f64>()
                + 2 * aabb.complex_buffer_size() * size_of::<c64>();
            let size = new_layer_size * node_size;
            println!("layer: {}, current_nodes: {}, new_layer_size: {}, extra_node: {}, current_slopes: {:?}, size: {}, {}", layer, current_nodes, new_layer_size, extra_node, current_slopes, size, human_readable_bytes(size));
            current_nodes = new_layer_size + extra_node as usize;
            layer += 1;
        }
    }

    #[test]
    fn test_stencil_size() {
        {
            println!("steps 10, 2d");
            let slopes = matrix![1, 1; 1, 1];
            let steps = 8000;
            test_tree_size(slopes, steps);
        }
    }
}
