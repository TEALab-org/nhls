use crate::domain::*;
use crate::fft_solver::*;
use crate::mem_fmt::*;
use crate::time_varying::*;
use crate::util::*;

// create layers, bottom up,
//
// note full ops?
//
// Maybe scan through, add to layer as we find them?
//
// We need two domain sized complex buffers, one domain sized real buffer
// We need two circstencils / complex buffers for all intermediate nodes
//
// So when we execute we first compute all circ stencil convolutions
// Then we move all of those into the complex domain
//
// Lets build ffts last? Make domain one root, full threads
// rest, divide layer by threads execute
//
pub const NONSENSE: usize = 99999999;

pub struct IRBase1Node {
    pub t: usize,
}

pub struct IRBase2Node<const GRID_DIMENSION: usize> {
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

pub struct IRConvolveNode<const GRID_DIMENSION: usize> {
    pub n1: usize,
    pub n2: usize,
    pub s_slopes: Bounds<GRID_DIMENSION>,
    pub s_size: usize,
    pub s1_offset: usize,
    pub s2_offset: usize,
    pub c_size: usize,
    pub cn: usize,
    pub c1_offset: usize,
    pub c2_offset: usize,
}

pub enum IRIntermediateNode<const GRID_DIMENSION: usize> {
    Base1(IRBase1Node),
    Base2(IRBase2Node<GRID_DIMENSION>),
    Convolve(IRConvolveNode<GRID_DIMENSION>),
}

pub struct TVPeriodicSolveBuilder<
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> {
    pub stencil: &'a StencilType,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
    pub aabb: AABB<GRID_DIMENSION>,
    pub nodes: Vec<Vec<IRIntermediateNode<GRID_DIMENSION>>>,
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    >
    TVPeriodicSolveBuilder<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
{
    pub fn new(stencil: &'a StencilType, aabb: AABB<GRID_DIMENSION>) -> Self {
        let stencil_slopes = stencil.slopes();

        TVPeriodicSolveBuilder {
            stencil,
            stencil_slopes,
            aabb,
            nodes: Vec::new(),
        }
    }

    pub fn add_node(
        &mut self,
        node: IRIntermediateNode<GRID_DIMENSION>,
        layer: usize,
    ) -> usize {
        while layer >= self.nodes.len() {
            self.nodes.push(Vec::new());
        }

        let result = self.nodes[layer].len();
        self.nodes[layer].push(node);
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
            return self.add_node(node, layer);
        }

        if end_time - start_time == 1 {
            let node = IRIntermediateNode::Base1(IRBase1Node { t: start_time });

            return self.add_node(node, layer);
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
            self.add_node(node, layer)
        }
    }

    pub fn build_solver(
        mut self,
        steps: usize,
        threads: usize,
        plan_type: PlanType,
    ) -> TVPeriodicSolver<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
    {
        let mut offset = 0;
        self.build_range(0, steps, 0, &mut offset);

        // Add complex buffers for solver
        let c_n = self.aabb.complex_buffer_size();
        let c_size = Self::complex_buffer_size(c_n);
        let c1_offset = offset;
        offset += c_size;
        let c2_offset = offset;
        offset += c_size;

        println!("Solve builder mem req: {}", human_readable_bytes(offset));
        let scratch = APScratch::new(offset);

        let mut fft_gen = FFTGen::new(plan_type);

        // Add whole domain op as 0
        fft_gen.get_op(self.aabb.exclusive_bounds(), threads);

        // Build FFT Solver
        println!("Solver Builder Report:");
        let mut result_nodes = Vec::with_capacity(self.nodes.len());
        for (layer, ir_layer_nodes) in self.nodes.iter().enumerate() {
            let mut layer_nodes = Vec::with_capacity(self.nodes[layer].len());
            let plan_threads = 1.max(
                (threads as f64 / layer_nodes.len() as f64).ceil() as usize,
            );
            println!(
                "Layer: {}, size: {}, plan_threads: {}",
                layer,
                layer_nodes.len(),
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
        let fft_plans = fft_gen.finish();

        let c_n = self.aabb.complex_buffer_size();
        let c1 = scratch.unsafe_get_buffer(c1_offset, c_size);
        let c2 = scratch.unsafe_get_buffer(c2_offset, c_size);
        let chunk_size = c_n / (2 * threads);

        TVPeriodicSolver {
            c1,
            c2,
            fft_plans,
            intermediate_nodes: result_nodes,
            chunk_size,
            aabb: self.aabb,
            stencil: self.stencil,
            threads,
            scratch,
        }
    }
}
