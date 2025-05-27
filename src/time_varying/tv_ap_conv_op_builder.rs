use crate::domain::*;
use crate::fft_solver::*;
use crate::mem_fmt::*;
use crate::stencil::*;
use crate::time_varying::*;
use crate::util::*;
use std::collections::HashMap;

pub struct TVIRBase1Node {
    pub t: usize,
}

pub struct TVIRBase2Node<const GRID_DIMENSION: usize> {
    pub t1: usize,
    pub t2: usize,
    pub s_slopes: Bounds<GRID_DIMENSION>,
    pub s_size: usize,
    pub s1_offset: usize,
    pub s2_offset: usize,
    pub c_size: usize,
    pub cn: usize,
    pub c1_offset: usize,
    pub c2_offset: usize,
}

pub struct TVIRConvolveNode<const GRID_DIMENSION: usize> {
    pub n1_key: (usize, usize),
    pub n2_key: (usize, usize),
    pub s_slopes: Bounds<GRID_DIMENSION>,
    pub s_size: usize,
    pub s1_offset: usize,
    pub s2_offset: usize,
    pub c_size: usize,
    pub cn: usize,
    pub c1_offset: usize,
    pub c2_offset: usize,
}

pub enum TVIRIntermediateNode<const GRID_DIMENSION: usize> {
    Base1(TVIRBase1Node),
    Base2(TVIRBase2Node<GRID_DIMENSION>),
    Convolve(TVIRConvolveNode<GRID_DIMENSION>),
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
    pub nodes: Vec<Vec<TVIRIntermediateNode<GRID_DIMENSION>>>,
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
        node: TVIRIntermediateNode<GRID_DIMENSION>,
        layer: usize,
    ) -> usize {
        while layer >= self.nodes.len() {
            self.nodes.push(Vec::new());
        }

        let result = self.nodes[layer].len();
        self.nodes[layer].push(node);
        self.node_map
            .insert((start_time, end_time), (layer, result));
        //println!("add node [{}, {})", start_time, end_time);
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

            let node = TVIRIntermediateNode::Base2(TVIRBase2Node {
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
            let node =
                TVIRIntermediateNode::Base1(TVIRBase1Node { t: start_time });
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

            let node = TVIRIntermediateNode::Convolve(TVIRConvolveNode {
                n1_key: (0, n1),
                n2_key: (0, n2),
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

    fn add_nodes_for_op(
        &mut self,
        op: &TVOpDescriptor<GRID_DIMENSION>,
        offset: &mut usize,
    ) {
        //println!("Add nodes for op, [{}, {})", op.step_min, op.step_max);
        let key = (op.step_min, op.step_max);

        if self.node_map.contains_key(&key) {
            return;
        }

        // Stack stores (range, id)
        let mut stack: Vec<((usize, usize), (usize, usize))> = Vec::new();
        let mut start = op.step_min;
        //let mut i = 0;
        while start < op.step_max {
            //println!("  ** start: {}, {}", start, op.step_max);
            let mut end = op.step_max;
            'inner: while end > start {
                //println!("  *** {}, {}", start, end);
                let range = (start, end);
                if let Some(v) = self.node_map.get(&range) {
                    //println!(" Found!");
                    start = end;
                    stack.push((range, *v));
                    break 'inner;
                } else if end - start == 1 {
                    //println!(" Add single");
                    // add node to base layer
                    let node =
                        TVIRIntermediateNode::Base1(TVIRBase1Node { t: start });
                    let layer = self.nodes.len() - 1;
                    let id = self.add_node(start, end, node, layer);
                    stack.push(((start, end), (layer, id)));
                    start = end;
                    break 'inner;
                } else {
                    //println!(" Not found");
                }
                //println!("   **** s: {:?}", stack.len());
                end -= 1;
            }
            //i += 1;
        }
        //println!("  - stack : {:?}", stack);

        assert!(stack.len() >= 2);

        let (mut result_node_range, mut result_node_id) = stack.last().unwrap();
        for i in (0..stack.len() - 1).rev() {
            let (new_node_range, new_node_id) = stack[i];

            let start_time = new_node_range.0;
            let end_time = result_node_range.1;
            //println!( " st: {}, et: {}, nnr: {:?}", start_time, end_time, new_node_range);
            debug_assert!(start_time <= end_time);

            let combined_slopes =
                (end_time - start_time) as i32 * self.stencil_slopes;
            let stencil_aabb = slopes_to_circ_aabb(&combined_slopes);
            //let mid = (start_time + end_time) / 2;
            // For full nodes we don't need to add anything, first layer is "summed"
            if stencil_aabb.ex_greater_than(&self.aabb) {
                panic!("Should never have full stencil whil adding op nodes");
            } else {
                assert!(new_node_id.0.min(result_node_id.0) >= 1);
                let next_layer = new_node_id.0.min(result_node_id.0) - 1;
                let mut n1_key = result_node_id;
                let mut n2_key = new_node_id;
                //println!("  PRE next_layer: {}, n1 l: {}, n2 l: {}", next_layer, n1_key.0, n2_key.0);

                // Indexing during solve is relative, as we slice up node layers
                n1_key.0 -= next_layer + 1;
                n2_key.0 -= next_layer + 1;

                //println!("  POST next_layer: {}, n1 l: {}, n2 l: {}", next_layer, n1_key.0, n2_key.0);

                let circ_aabb = slopes_to_circ_aabb(&combined_slopes);
                let domain_size =
                    Self::real_domain_size(circ_aabb.buffer_size());
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

                let node = TVIRIntermediateNode::Convolve(TVIRConvolveNode {
                    n1_key,
                    n2_key,
                    s_slopes: combined_slopes,
                    s_size: domain_size,
                    s1_offset,
                    s2_offset,
                    c_size: complex_size,
                    cn: c_n,
                    c1_offset,
                    c2_offset,
                });

                let id = self.add_node(start_time, end_time, node, next_layer);
                result_node_id = (next_layer, id);
                result_node_range = (start_time, end_time);
            }
        }
    }

    fn add_op_nodes(
        &mut self,
        tree_queries: &[TVOpDescriptor<GRID_DIMENSION>],
        offset: &mut usize,
    ) {
        for query in tree_queries.iter() {
            self.add_nodes_for_op(query, offset);
        }
    }

    pub fn build_op_calc(
        mut self,
        steps: usize,
        threads: usize,
        plan_type: PlanType,
        tree_queries: &[TVOpDescriptor<GRID_DIMENSION>],
    ) -> TVAPConvOpsCalc<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
    {
        // Calculate scratch space, build IR nodes
        let mut offset = 0;
        self.build_range(0, steps, 10, &mut offset);
        let node_count: usize = self.nodes.iter().map(|ns| ns.len()).sum();
        println!(
            "Solve builder pre mem req: {}, nodes: {}",
            human_readable_bytes(offset),
            node_count
        );

        self.add_op_nodes(tree_queries, &mut offset);

        let node_count: usize = self.nodes.iter().map(|ns| ns.len()).sum();
        println!(
            "Solve builder after op nodes mem req: {}, nodes: {}",
            human_readable_bytes(offset),
            node_count
        );

        // TODO make sure we add nodes for missing op stencils
        let scratch = APScratch::new(offset);
        /*
        println!("NODE MAP");
        for (k, j) in &self.node_map {
            println!("{:?} -> {:?}", k, j);
        }
        */

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
                    TVIRIntermediateNode::Base1(n) => {
                        layer_nodes.push(TVIntermediateNode::Base1(
                            TVBase1Node { t: n.t },
                        ));
                    }
                    TVIRIntermediateNode::Base2(n) => {
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
                        layer_nodes.push(TVIntermediateNode::Base2(
                            TVBase2Node {
                                t1: n.t1,
                                t2: n.t2,
                                s1,
                                s2,
                                c1,
                                c2,
                                plan_id,
                            },
                        ));
                    }
                    TVIRIntermediateNode::Convolve(n) => {
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
                        layer_nodes.push(TVIntermediateNode::Convolve(
                            TVConvolveNode {
                                n1_key: n.n1_key,
                                n2_key: n.n2_key,
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
        for op_descriptor in tree_queries.iter() {
            // Generate fft op, create ConvOp
            let fft_pair_id = fft_gen
                .get_op(op_descriptor.exclusive_bounds, op_descriptor.threads);

            let node_key = (op_descriptor.step_min, op_descriptor.step_max);
            //println!("node_key: {:?}", node_key);
            let op = ConvOp {
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
