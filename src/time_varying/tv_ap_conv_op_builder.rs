use crate::domain::*;
use crate::fft_solver::*;
use crate::mem_fmt::*;
use crate::time_varying::*;
use crate::util::*;
use std::collections::HashMap;

struct OpOffsets {
    real_domain_offset: usize,
    real_domain_size: usize,
    op_buffer_offset: usize,
    op_buffer_size: usize,
}

pub struct TVAPOpCalcBuilder<
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> {
    pub stencil: &'a StencilType,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
    pub aabb: AABB<GRID_DIMENSION>,
    pub nodes: Vec<Vec<IRIntermediateNode<GRID_DIMENSION>>>,
    // map (start_time, end_time) -> (layer, node_id)
    pub node_map: HashMap<(usize, usize), (usize, usize)>,
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    > TVAPOpCalcBuilder<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
{
    pub fn new(stencil: &'a StencilType, aabb: AABB<GRID_DIMENSION>) -> Self {
        let stencil_slopes = stencil.slopes();

        TVAPOpCalcBuilder {
            stencil,
            stencil_slopes,
            aabb,
            nodes: Vec::new(),
            node_map: HashMap::new(),
        }
    }

    pub fn add_node(
        &mut self,
        start_time: usize,
        end_time: usize,
        node: IRIntermediateNode<GRID_DIMENSION>,
        layer: usize,
    ) -> usize {
        while layer >= self.nodes.len() {
            self.nodes.push(Vec::new());
        }

        let result = self.nodes[layer].len();
        self.nodes[layer].push(node);
        self.node_map
            .insert((start_time, end_time), (layer, result));
        result
    }

    /// Size for real domain in bytes respecting MIN_ALIGNMENT
    pub fn real_domain_size(n_reals: usize) -> usize {
        let min_bytes = n_reals * std::mem::size_of::<f64>();
        let blocks = min_bytes.div_ceil(MIN_ALIGNMENT);
        blocks * MIN_ALIGNMENT
    }

    /// Size for complex buffer in bytes respecting MIN_ALIGNMENT
    pub fn complex_buffer_size(n_c: usize) -> usize {
        let min_bytes = n_c * std::mem::size_of::<c64>();
        let blocks = min_bytes.div_ceil(MIN_ALIGNMENT);
        blocks * MIN_ALIGNMENT
    }

    pub fn build_range(
        &mut self,
        start_time: usize,
        end_time: usize,
        layer: usize,
        offset: &mut usize,
    ) -> usize {
        debug_assert!(end_time > start_time);

        // Two Base cases is combining to single step stencils
        if end_time - start_time == 2 {
            let combined_slopes = self.stencil_slopes * 2;
            let circ_aabb = slopes_to_circ_aabb(&combined_slopes);
            let domain_size = Self::real_domain_size(circ_aabb.buffer_size());
            let c_n = circ_aabb.complex_buffer_size();
            let complex_size = Self::complex_buffer_size(c_n);
            let s1_offset = *offset;
            *offset += domain_size;
            let s2_offset = *offset;
            *offset += domain_size;
            let c1_offset = *offset;
            *offset += complex_size;
            let c2_offset = *offset;
            *offset += complex_size;

            let node = IRIntermediateNode::Base2(IRBase2Node {
                t1: start_time,
                t2: start_time + 1,
                s_slopes: combined_slopes,
                s_size: domain_size,
                s1_offset,
                s2_offset,
                c_size: complex_size,
                cn: c_n,
                c1_offset,
                c2_offset,
            });
            return self.add_node(start_time, end_time, node, layer);
        }

        if end_time - start_time == 1 {
            let node = IRIntermediateNode::Base1(IRBase1Node { t: start_time });
            return self.add_node(start_time, end_time, node, layer);
        }

        let combined_slopes =
            (end_time - start_time) as i32 * self.stencil_slopes;
        let stencil_aabb = slopes_to_circ_aabb(&combined_slopes);
        let mid = (start_time + end_time) / 2;
        // For full nodes we don't need to add anything, first layer is "summed"
        if stencil_aabb.ex_greater_than(&self.aabb) {
            self.build_range(start_time, mid, layer, offset);
            self.build_range(mid, end_time, layer, offset);
            NONSENSE
        } else {
            let n1 = self.build_range(start_time, mid, layer + 1, offset);
            let n2 = self.build_range(mid, end_time, layer + 1, offset);
            let circ_aabb = slopes_to_circ_aabb(&combined_slopes);
            let domain_size = Self::real_domain_size(circ_aabb.buffer_size());
            let c_n = circ_aabb.complex_buffer_size();
            let complex_size = Self::complex_buffer_size(c_n);
            let s1_offset = *offset;
            *offset += domain_size;
            let s2_offset = *offset;
            *offset += domain_size;
            let c1_offset = *offset;
            *offset += complex_size;
            let c2_offset = *offset;
            *offset += complex_size;

            let node = IRIntermediateNode::Convolve(IRConvolveNode {
                n1,
                n2,
                s_slopes: combined_slopes,
                s_size: domain_size,
                s1_offset,
                s2_offset,
                c_size: complex_size,
                cn: c_n,
                c1_offset,
                c2_offset,
            });
            self.add_node(start_time, end_time, node, layer)
        }
    }

    fn build_op_offsets(
        &self,
        tree_queries: &[TVOpDescriptor<GRID_DIMENSION>],
        offset: &mut usize,
    ) -> Vec<OpOffsets> {
        // Add complex buffers for solver
        let mut op_offsets = Vec::with_capacity(tree_queries.len());
        for op in tree_queries.iter() {
            let query_aabb = AABB::from_exclusive_bounds(&op.exclusive_bounds);
            // real domain,
            let real_domain_offset = *offset;
            let real_domain_size =
                Self::real_domain_size(query_aabb.buffer_size());
            *offset += real_domain_size;

            // op_buffer,
            let op_buffer_offset = *offset;
            let op_buffer_size =
                Self::complex_buffer_size(query_aabb.complex_buffer_size());
            let op_offset = OpOffsets {
                real_domain_offset,
                real_domain_size,
                op_buffer_offset,
                op_buffer_size,
            };

            op_offsets.push(op_offset);
        }
        op_offsets
    }

    pub fn build_op_calc(
        mut self,
        steps: usize,
        threads: usize,
        plan_type: PlanType,
        tree_queries: &Vec<TVOpDescriptor<GRID_DIMENSION>>,
    ) -> TVAPConvOpsCalc<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
    {
        // Calculate scratch space, build IR nodes
        let mut offset = 0;
        self.build_range(0, steps, 0, &mut offset);
        let op_offsets = self.build_op_offsets(tree_queries, &mut offset);
        println!("Solve builder mem req: {}", human_readable_bytes(offset));
        let scratch = APScratch::new(offset);

        // Use TVOpDescriptors
        // tree_qieru add fft ops with threads
        // Don't need central solve
        // fft_gen.get_op(self.aabb.exclusive_bounds(), threads);
        // I guess I want to do this here so that we get the solver threads right?
        // fft store should really do both threads and size, but w/e for now
        //
        // At any rate,
        // we allso need to account for scratch for conv ops
        let mut fft_gen = FFTGen::new(plan_type);

        // Build FFT Solver
        println!("Solver Builder Report:");
        let mut result_nodes = Vec::with_capacity(self.nodes.len());
        for (layer, ir_layer_nodes) in self.nodes.iter().enumerate() {
            let mut layer_nodes = Vec::with_capacity(self.nodes[layer].len());
            let plan_threads = 1
                .max((threads as f64 / ir_layer_nodes.len() as f64).ceil()
                    as usize);
            println!(
                "Layer: {}, size: {}, plan_threads: {}",
                layer,
                ir_layer_nodes.len(),
                plan_threads
            );
            for node in ir_layer_nodes.iter() {
                match node {
                    IRIntermediateNode::Base1(n) => {
                        layer_nodes.push(IntermediateNode::Base1(Base1Node {
                            t: n.t,
                        }));
                    }
                    IRIntermediateNode::Base2(n) => {
                        let s1_buffer =
                            scratch.unsafe_get_buffer(n.s1_offset, n.s_size);
                        let s2_buffer =
                            scratch.unsafe_get_buffer(n.s2_offset, n.s_size);

                        let s1 = CircStencil::new(n.s_slopes, s1_buffer);
                        let s2 = CircStencil::new(n.s_slopes, s2_buffer);

                        let c1 = &mut scratch
                            .unsafe_get_buffer(n.c1_offset, n.c_size)
                            [0..n.cn];
                        let c2 = &mut scratch
                            .unsafe_get_buffer(n.c2_offset, n.c_size)
                            [0..n.cn];

                        let size = s1.domain.aabb().exclusive_bounds();
                        let plan_id = fft_gen.get_op(size, plan_threads);
                        layer_nodes.push(IntermediateNode::Base2(Base2Node {
                            t1: n.t1,
                            t2: n.t2,
                            s1,
                            s2,
                            c1,
                            c2,
                            plan_id,
                        }));
                    }
                    IRIntermediateNode::Convolve(n) => {
                        let s1_buffer =
                            scratch.unsafe_get_buffer(n.s1_offset, n.s_size);
                        let s2_buffer =
                            scratch.unsafe_get_buffer(n.s2_offset, n.s_size);

                        let s1 = CircStencil::new(n.s_slopes, s1_buffer);
                        let s2 = CircStencil::new(n.s_slopes, s2_buffer);

                        let c1 = &mut scratch
                            .unsafe_get_buffer(n.c1_offset, n.c_size)
                            [0..n.cn];
                        let c2 = &mut scratch
                            .unsafe_get_buffer(n.c2_offset, n.c_size)
                            [0..n.cn];

                        let size = s1.domain.aabb().exclusive_bounds();
                        let plan_id = fft_gen.get_op(size, plan_threads);
                        layer_nodes.push(IntermediateNode::Convolve(
                            ConvolveNode {
                                n1: n.n1,
                                n2: n.n2,
                                s1,
                                s2,
                                c1,
                                c2,
                                plan_id,
                            },
                        ));
                    }
                }
            }
            result_nodes.push(layer_nodes);
        }

        // Build ConvOp intsances
        // TODO we need the node lookip
        let mut conv_ops = Vec::with_capacity(tree_queries.len());
        for (op_descriptor, op_offset) in
            tree_queries.iter().zip(op_offsets.iter())
        {
            // Generate fft op, create ConvOp
            let fft_pair_id = fft_gen
                .get_op(op_descriptor.exclusive_bounds, op_descriptor.threads);

            let real_domain = scratch.unsafe_get_buffer(
                op_offset.real_domain_offset,
                op_offset.real_domain_size,
            );

            let op_buffer = scratch.unsafe_get_buffer(
                op_offset.op_buffer_offset,
                op_offset.op_buffer_size,
            );

            let node_key = (op_descriptor.step_min, op_descriptor.step_max);

            let op = ConvOp {
                real_domain: SliceDomain::new(
                    AABB::from_exclusive_bounds(
                        &op_descriptor.exclusive_bounds,
                    ),
                    real_domain,
                ),
                op_buffer,
                fft_pair_id,
                node: *self.node_map.get(&node_key).unwrap(),
            };
            conv_ops.push(op);
        }

        let fft_pairs = fft_gen.finish();

        TVAPConvOpsCalc {
            stencil: self.stencil,
            aabb: self.aabb,
            intermediate_nodes: result_nodes,
            conv_ops,
            threads,
            fft_pairs,
            scratch,
        }
    }
}
