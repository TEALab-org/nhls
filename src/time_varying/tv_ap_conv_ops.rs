use crate::domain::*;
use crate::fft_solver::*;
use crate::par_slice;
use crate::time_varying::*;
use crate::util::*;
use fftw::plan::*;
use rayon::prelude::*;

pub struct ConvOp<'a, const GRID_DIMENSION: usize> {
    pub real_domain: SliceDomain<'a, GRID_DIMENSION>,
    pub op_buffer: &'a mut [c64],
    pub fft_pair_id: FFTPairId,
    pub node: (usize, usize),
}

pub fn build_op<
    'solver_life,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
>(
    op: &mut ConvOp<'solver_life, GRID_DIMENSION>,
    stencil: &StencilType,
    intermediate_nodes: &[Vec<
        IntermediateNode<'solver_life, GRID_DIMENSION>,
    >],
    aabb: &AABB<GRID_DIMENSION>,
    fft_store: &FFTStore,
    global_time: usize,
) {
    let ir_node = &intermediate_nodes[op.node.0][op.node.1];

    // Add weights
    match ir_node {
        IntermediateNode::Base1(n) => {
            // Add stencil offsets to output
            let weights = stencil.weights(global_time + n.t);
            for i in 0..NEIGHBORHOOD_SIZE {
                let offset = stencil.offsets()[i];
                let weight = weights[i];
                let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                let periodic_coord = aabb.periodic_coord(&rn_i);
                op.real_domain.set_coord(&periodic_coord, weight);
            }
        }
        IntermediateNode::Base2(n) => {
            // Add stencil offsets to output
            for (offset, weight) in n.s1.to_offset_weights() {
                let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                let periodic_coord = aabb.periodic_coord(&rn_i);
                op.real_domain.set_coord(&periodic_coord, weight);
            }
        }
        IntermediateNode::Convolve(n) => {
            // Add stencil offsets to output
            for (offset, weight) in n.s1.to_offset_weights() {
                let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                let periodic_coord = aabb.periodic_coord(&rn_i);
                op.real_domain.set_coord(&periodic_coord, weight);
            }
        }
    }

    // fft stuff
    fft_store
        .get(op.fft_pair_id)
        .forward_plan
        .r2c(op.real_domain.buffer_mut(), op.op_buffer)
        .unwrap();

    // Cleanup
    match ir_node {
        IntermediateNode::Base1(_) => {
            // Add stencil offsets to output
            for i in 0..NEIGHBORHOOD_SIZE {
                let offset = stencil.offsets()[i];
                let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                let periodic_coord = aabb.periodic_coord(&rn_i);
                op.real_domain.set_coord(&periodic_coord, 0.0);
            }
        }
        IntermediateNode::Base2(n) => {
            // Add stencil offsets to output
            for (offset, _weight) in n.s1.to_offset_weights() {
                let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                let periodic_coord = aabb.periodic_coord(&rn_i);
                op.real_domain.set_coord(&periodic_coord, 0.0);
            }
        }
        IntermediateNode::Convolve(n) => {
            // Add stencil offsets to output
            for (offset, _weight) in n.s1.to_offset_weights() {
                let rn_i: Coord<GRID_DIMENSION> = offset * -1;
                let periodic_coord = aabb.periodic_coord(&rn_i);
                op.real_domain.set_coord(&periodic_coord, 0.0);
            }
        }
    }
}

pub struct TVAPConvOpsCalc<
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> {
    pub stencil: &'a StencilType,
    pub aabb: AABB<GRID_DIMENSION>,
    pub intermediate_nodes: Vec<Vec<IntermediateNode<'a, GRID_DIMENSION>>>,
    pub conv_ops: Vec<ConvOp<'a, GRID_DIMENSION>>,
    pub threads: usize,
    pub fft_pairs: FFTStore,
    pub scratch: APScratch,
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    > TVAPConvOpsCalc<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
{
    pub fn build_ops(&mut self, global_time: usize) {
        // Build Tree like periodic solver
        println!("Solver: Build base layer");
        // Build intermediate tree layers
        let base_layer_id = self.intermediate_nodes.len() - 1;
        solve_base_layer(
            self.stencil,
            self.threads,
            global_time,
            &mut self.intermediate_nodes[base_layer_id],
            &self.fft_pairs,
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
                &self.fft_pairs,
            );
        }

        // Build ops
        println!("Solver: Build Ops");
        self.conv_ops.par_iter_mut().for_each(|conv_op| {
            build_op(
                conv_op,
                self.stencil,
                &self.intermediate_nodes,
                &self.aabb,
                &self.fft_pairs,
                global_time,
            );
        });
    }

    /// And input / output / complex buffers
    pub fn apply_convolution<DomainType: DomainView<GRID_DIMENSION>>(
        &self,
        id: usize,
        input: &mut DomainType,
        output: &mut DomainType,
        complex_buffer: &mut [c64],
        chunk_size: usize,
    ) {
        let op = &self.conv_ops[id];
        let fft_pair = self.fft_pairs.get(op.fft_pair_id);
        let n_r = input.aabb().buffer_size();
        let n_c = input.aabb().complex_buffer_size();
        fft_pair
            .forward_plan
            .r2c(input.buffer_mut(), &mut complex_buffer[0..n_c])
            .unwrap();
        par_slice::multiply_by(
            &mut complex_buffer[0..n_c],
            op.op_buffer,
            chunk_size,
        );
        fft_pair
            .backward_plan
            .c2r(&mut complex_buffer[0..n_c], output.buffer_mut())
            .unwrap();
        par_slice::div(output.buffer_mut(), n_r as f64, chunk_size);
    }
}
