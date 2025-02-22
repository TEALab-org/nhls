use crate::domain::*;
use crate::fft_solver::*;
use crate::time_varying::*;
use crate::util::*;
use std::io::prelude::*;

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

pub struct Base1Node {
    pub t: usize,
}

pub struct Base2Node<const GRID_DIMENSION: usize> {
    pub t1: usize,
    pub t2: usize,
    pub s1: CircStencil<GRID_DIMENSION>,
    pub s2: CircStencil<GRID_DIMENSION>,
    pub c1: AlignedVec<c64>,
    pub c2: AlignedVec<c64>,
    pub plan_id: FFTPairId,
}

pub struct ConvolveNode<const GRID_DIMENSION: usize> {
    pub n1: usize,
    pub n2: usize,
    pub s1: CircStencil<GRID_DIMENSION>,
    pub s2: CircStencil<GRID_DIMENSION>,
    pub c1: AlignedVec<c64>,
    pub c2: AlignedVec<c64>,
    pub plan_id: FFTPairId,
}

pub enum IntermediateNode<const GRID_DIMENSION: usize> {
    Base1(Base1Node),
    Base2(Base2Node<GRID_DIMENSION>),
    Convolve(ConvolveNode<GRID_DIMENSION>),
}

impl<const GRID_DIMENSION: usize> IntermediateNode<GRID_DIMENSION> {
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

pub struct TVPeriodicSolveBuilder<
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> {
    pub stencil: &'a StencilType,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
    pub aabb: AABB<GRID_DIMENSION>,
    pub nodes: Vec<Vec<IntermediateNode<GRID_DIMENSION>>>,
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
        node: IntermediateNode<GRID_DIMENSION>,
        layer: usize,
    ) -> TVNodeId {
        while layer >= self.nodes.len() {
            self.nodes.push(Vec::new());
        }

        let result = self.nodes[layer].len();
        self.nodes[layer].push(node);
        result
    }

    pub fn build_range(
        &mut self,
        start_time: usize,
        end_time: usize,
        layer: usize,
    ) -> TVNodeId {
        debug_assert!(end_time > start_time);

        // Two Base cases is combining to single step stencils
        if end_time - start_time == 2 {
            let combined_slopes = self.stencil_slopes * 2;
            let s1 = CircStencil::new(combined_slopes);
            let s2 = CircStencil::new(combined_slopes);
            let c_n = s1.domain.aabb().complex_buffer_size();
            let c1 = AlignedVec::new(c_n);
            let c2 = AlignedVec::new(c_n);

            let node = IntermediateNode::Base2(Base2Node {
                t1: start_time,
                t2: start_time + 1,
                s1,
                s2,
                c1,
                c2,
                plan_id: NONSENSE,
            });
            return self.add_node(node, layer);
        }

        if end_time - start_time == 1 {
            let node = IntermediateNode::Base1(Base1Node { t: start_time });

            return self.add_node(node, layer);
        }

        let combined_slopes =
            (end_time - start_time) as i32 * self.stencil_slopes;
        let stencil_aabb = slopes_to_circ_aabb(&combined_slopes);
        let mid = (start_time + end_time) / 2;
        // For full nodes we don't need to add anything, first layer is "summed"
        if stencil_aabb.ex_greater_than(&self.aabb) {
            self.build_range(start_time, mid, layer);
            self.build_range(mid, end_time, layer);
            NONSENSE
        } else {
            let n1 = self.build_range(start_time, mid, layer + 1);
            let n2 = self.build_range(mid, end_time, layer + 1);
            let s1 = CircStencil::new(combined_slopes);
            let s2 = CircStencil::new(combined_slopes);
            let c_n = s1.domain.aabb().complex_buffer_size();
            let c1 = AlignedVec::new(c_n);
            let c2 = AlignedVec::new(c_n);

            let node = IntermediateNode::Convolve(ConvolveNode {
                n1,
                n2,
                s1,
                s2,
                c1,
                c2,
                plan_id: NONSENSE,
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
        self.build_range(0, steps, 0);
        let mut fft_gen = FFTGen::new(plan_type);

        // Add whole domain op as 0
        fft_gen.get_op(self.aabb.exclusive_bounds(), threads);

        // Build FFT Solver
        for layer_nodes in self.nodes.iter_mut() {
            let plan_threads = 1.max(threads / layer_nodes.len());
            /*
            println!(
                "Layer: {}, size: {}, plan_threads: {}",
                layer,
                layer_nodes.len(),
                plan_threads
            );
            */
            for node in layer_nodes.iter_mut() {
                match node {
                    IntermediateNode::Base1(_) => {}
                    IntermediateNode::Base2(n) => {
                        let size = n.s1.domain.aabb().exclusive_bounds();
                        let plan_id = fft_gen.get_op(size, plan_threads);
                        n.plan_id = plan_id;
                    }
                    IntermediateNode::Convolve(n) => {
                        let size = n.s1.domain.aabb().exclusive_bounds();
                        let plan_id = fft_gen.get_op(size, plan_threads);
                        n.plan_id = plan_id;
                    }
                }
            }
        }
        let fft_plans = fft_gen.finish();

        let c_n = self.aabb.complex_buffer_size();
        let c1 = AlignedVec::new(c_n);
        let c2 = AlignedVec::new(c_n);
        let chunk_size = c_n / (2 * threads);

        TVPeriodicSolver {
            c1,
            c2,
            fft_plans,
            intermediate_nodes: self.nodes,
            chunk_size,
            aabb: self.aabb,
            stencil: self.stencil,
            threads,
        }
    }

    pub fn to_debug_file<P: AsRef<std::path::Path>>(&self, path: &P) {
        println!(
            "Writing periodic_solver_builder_debug_file: {:?}",
            path.as_ref()
        );
        let mut w =
            std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        for (l, layer_nodes) in self.nodes.iter().enumerate() {
            for (i, node) in layer_nodes.iter().enumerate() {
                match node {
                    IntermediateNode::Base1(b1) => {
                        writeln!(w, "l: {}, i: {}, Base1, t: {}", l, i, b1.t)
                            .unwrap();
                    }
                    IntermediateNode::Base2(b2) => {
                        writeln!(
                            w,
                            "l: {}, i: {}, Base2, t1: {}, t2: {}, {}",
                            l,
                            i,
                            b2.t1,
                            b2.t2,
                            b2.s2.domain.aabb()
                        )
                        .unwrap();
                    }
                    IntermediateNode::Convolve(c) => {
                        writeln!(
                            w,
                            "l: {}, i: {}, Convolve , n1: {}, n2: {}, {}",
                            l,
                            i,
                            c.n1,
                            c.n2,
                            c.s2.domain.aabb()
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
}
