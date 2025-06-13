use crate::ap_solver::index_types::*;
use crate::ap_solver::periodic_ops::*;
use crate::ap_solver::scratch::*;
use crate::domain::*;
use crate::par_slice;
use crate::stencil::*;
use crate::time_varying::{FFTPairId, FFTStore};
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
    pub n1_key: (usize, usize),
    pub n2_key: (usize, usize),
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

impl<const GRID_DIMENSION: usize> IntermediateNode<'_, GRID_DIMENSION> {
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

pub struct ConvOp {
    pub fft_pair_id: FFTPairId,
    pub node: (usize, usize),
}

pub fn solve_tvbase2_node<
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
    par_slice::multiply_by(node.c1, node.c2, chunk_size);
    fft_pair
        .backward_plan
        .c2r(node.c1, node.s1.domain.buffer_mut())
        .unwrap();
    let n_r = node.s1.domain.aabb().buffer_size();
    par_slice::div(node.s1.domain.buffer_mut(), n_r as f64, chunk_size);
}

/// Helper function for solve_convolve_node
pub fn add_tvnode_to_circ_stencil<
    'b,
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
>(
    s: &'b mut CircStencil<'a, GRID_DIMENSION>,
    id: (usize, usize),
    prev_layers: &'b [Vec<IntermediateNode<'a, GRID_DIMENSION>>],
    stencil: &StencilType,
    global_time: usize,
) {
    match &prev_layers[id.0][id.1] {
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

pub fn solve_tvconvolve_node<
    'solver_life,
    'node_borrow,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
>(
    node: &'node_borrow mut ConvolveNode<'solver_life, GRID_DIMENSION>,
    prev_layers: &'node_borrow [Vec<
        IntermediateNode<'solver_life, GRID_DIMENSION>,
    >],
    stencil: &'node_borrow StencilType,
    global_time: usize,
    fft_store: &FFTStore,
    chunk_size: usize,
) {
    add_tvnode_to_circ_stencil(
        &mut node.s1,
        node.n1_key,
        prev_layers,
        stencil,
        global_time,
    );
    add_tvnode_to_circ_stencil(
        &mut node.s2,
        node.n2_key,
        prev_layers,
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

pub fn solve_tvbase_layer<
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
                        solve_tvbase2_node(
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

pub fn solve_tvmiddle_layer<
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
    prev_layers: &'node_borrow [Vec<
        IntermediateNode<'solver_life, GRID_DIMENSION>,
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
                        solve_tvbase2_node(
                            n,
                            stencil,
                            global_time,
                            fft_store,
                            chunk_size,
                        );
                    }
                    IntermediateNode::Convolve(n) => {
                        solve_tvconvolve_node(
                            n,
                            prev_layers,
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

pub struct TvPeriodicOps<
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> {
    pub stencil: &'a StencilType,
    pub aabb: AABB<GRID_DIMENSION>,
    pub intermediate_nodes: Vec<Vec<IntermediateNode<'a, GRID_DIMENSION>>>,
    pub conv_ops: Vec<ConvOp>,
    pub threads: usize,
    pub fft_pairs: FFTStore,
    pub scratch: Scratch,
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    > TvPeriodicOps<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
{
    pub fn blank(s: &'a StencilType) -> Self {
        TvPeriodicOps {
            stencil: s,
            aabb: AABB::new(Bounds::zero()),
            intermediate_nodes: Vec::new(),
            conv_ops: Vec::new(),
            threads: 0,
            fft_pairs: FFTStore::new(Vec::new()),
            scratch: Scratch::new(1),
        }
    }

    pub fn build_ops(&mut self, global_time: usize) {
        // Build Tree like periodic solver
        println!("Solver: Build base layer");
        for layer in self.intermediate_nodes.iter_mut() {
            layer.par_iter_mut().for_each(|n| n.clear_stencils());
        }

        // Build intermediate tree layers
        let base_layer_id = self.intermediate_nodes.len() - 1;
        solve_tvbase_layer(
            self.stencil,
            self.threads,
            global_time,
            &mut self.intermediate_nodes[base_layer_id],
            &self.fft_pairs,
        );
        for layer_id in (0..base_layer_id).rev() {
            println!("Solver: build layer: {}", layer_id);
            let (new, old) = self.intermediate_nodes.split_at_mut(layer_id + 1);
            solve_tvmiddle_layer(
                self.stencil,
                self.threads,
                global_time,
                new.last_mut().unwrap(),
                &old,
                &self.fft_pairs,
            );
        }
    }

    /// And input / output / complex buffers
    pub fn apply_convolution<DomainType: DomainView<GRID_DIMENSION>>(
        &self,
        id: usize,
        input: &mut DomainType,
        output: &mut DomainType,
        domain_complex_buffer: &mut [c64],
        op_complex_buffer: &mut [c64],
        chunk_size: usize,
        central_global_time: usize,
    ) {
        let op = &self.conv_ops[id];
        let ir_node = &self.intermediate_nodes[op.node.0][op.node.1];
        par_slice::set_value(output.buffer_mut(), 0.0, chunk_size);
        let n_r = input.aabb().buffer_size();
        let n_c = input.aabb().complex_buffer_size();
        let mut s_d = output.unsafe_mut_access();
        s_d.set_aabb(AABB::from_exclusive_bounds(
            &output.aabb().exclusive_bounds(),
        ));

        // Add weights
        match ir_node {
            IntermediateNode::Base1(n) => {
                // Add stencil offsets to output
                let weights = self.stencil.weights(central_global_time + n.t);
                for i in 0..NEIGHBORHOOD_SIZE {
                    let offset = self.stencil.offsets()[i];
                    let weight = weights[i];
                    let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                    let periodic_coord = s_d.aabb().periodic_coord(&rn_i);
                    s_d.set_coord(&periodic_coord, weight);
                }
            }
            IntermediateNode::Base2(n) => {
                // Add stencil offsets to output
                for (offset, weight) in n.s1.to_offset_weights() {
                    let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                    let periodic_coord = s_d.aabb().periodic_coord(&rn_i);
                    s_d.set_coord(&periodic_coord, weight);
                }
            }
            IntermediateNode::Convolve(n) => {
                // Add stencil offsets to output
                for (offset, weight) in n.s1.to_offset_weights() {
                    let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                    let periodic_coord = s_d.aabb().periodic_coord(&rn_i);
                    s_d.set_coord(&periodic_coord, weight);
                }
            }
        }

        // fft stuff
        self.fft_pairs
            .get(op.fft_pair_id)
            .forward_plan
            .r2c(s_d.buffer_mut(), &mut op_complex_buffer[0..n_c])
            .unwrap();

        let fft_pair = self.fft_pairs.get(op.fft_pair_id);
        fft_pair
            .forward_plan
            .r2c(input.buffer_mut(), &mut domain_complex_buffer[0..n_c])
            .unwrap();

        // We now need to build the op here
        // We can use output buffer for stencil circulent
        // we can use extra complex buffer for that

        par_slice::multiply_by(
            &mut domain_complex_buffer[0..n_c],
            &op_complex_buffer[0..n_c],
            chunk_size,
        );
        fft_pair
            .backward_plan
            .c2r(&mut domain_complex_buffer[0..n_c], output.buffer_mut())
            .unwrap();
        par_slice::div(output.buffer_mut(), n_r as f64, chunk_size);
    }
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    > PeriodicOps<GRID_DIMENSION>
    for TvPeriodicOps<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
{
    fn build_ops(&mut self, global_time: usize) {
        self.build_ops(global_time);
    }

    fn apply_operation<'b>(
        &self,
        op_id: OpId,
        input: &mut SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
        // NOTE: TV will need twice the buffer, and we can split it as needed
        complex_buffer: &mut [c64],
        central_global_time: usize,
        chunk_size: usize,
    ) {
        let (domain_complex_buffer, op_complex_buffer) =
            complex_buffer.split_at_mut(complex_buffer.len() / 2);
        self.apply_convolution(
            op_id,
            input,
            output,
            domain_complex_buffer,
            op_complex_buffer,
            chunk_size,
            central_global_time,
        )
    }
}
