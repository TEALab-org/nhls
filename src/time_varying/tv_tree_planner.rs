use crate::domain::*;
use crate::time_varying::*;
use crate::util::*;
use fftw::array::*;
use std::collections::HashMap;
use std::io::prelude::*;

pub type TVNodeId = usize;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TVTreePlanType {
    Base1,
    Base2,
    Convolve,
    Full,
}

pub struct TVTreePlanNode<const GRID_DIMENSION: usize> {
    layer: usize,
    start_time: usize,
    end_time: usize,
    stencil_aabb: AABB<GRID_DIMENSION>,
    n_type: TVTreePlanType,
    sub_ops: Option<[TVNodeId; 2]>,
}

pub struct TVTreePlanner<
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> {
    pub stencil: &'a StencilType,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
    pub aabb: AABB<GRID_DIMENSION>,
    pub nodes: Vec<TVTreePlanNode<GRID_DIMENSION>>,
    pub max_layer: usize,
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    > TVTreePlanner<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
{
    pub fn new(stencil: &'a StencilType, aabb: AABB<GRID_DIMENSION>) -> Self {
        let stencil_slopes = stencil.slopes();

        TVTreePlanner {
            stencil,
            stencil_slopes,
            aabb,
            nodes: Vec::new(),
            max_layer: 0,
        }
    }

    pub fn add_node(
        &mut self,
        node: TVTreePlanNode<GRID_DIMENSION>,
    ) -> TVNodeId {
        let result = self.nodes.len();
        self.max_layer = self.max_layer.max(node.layer);
        self.nodes.push(node);
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
            let stencil_aabb = slopes_to_circ_aabb(&combined_slopes);

            let node = TVTreePlanNode {
                layer,
                start_time,
                end_time,
                stencil_aabb,
                n_type: TVTreePlanType::Base2,
                sub_ops: None,
            };
            return self.add_node(node);
        }

        // Edgecase, not preferable
        if end_time - start_time == 1 {
            let stencil_aabb = slopes_to_circ_aabb(&self.stencil_slopes);

            let node = TVTreePlanNode {
                layer,
                start_time,
                end_time,
                stencil_aabb,
                n_type: TVTreePlanType::Base1,
                sub_ops: None,
            };
            return self.add_node(node);
        }

        let combined_slopes =
            (end_time - start_time) as i32 * self.stencil_slopes;
        let mut stencil_aabb = slopes_to_circ_aabb(&combined_slopes);
        let mut n_type = TVTreePlanType::Convolve;
        if stencil_aabb.ex_greater_than(&self.aabb) {
            n_type = TVTreePlanType::Full;
            stencil_aabb = self.aabb;
        }

        let mid = (start_time + end_time) / 2;
        let n1 = self.build_range(start_time, mid, layer + 1);
        let n2 = self.build_range(mid, end_time, layer + 1);

        let node = TVTreePlanNode {
            layer,
            start_time,
            end_time,
            stencil_aabb,
            n_type,
            sub_ops: Some([n1, n2]),
        };
        self.add_node(node)
    }

    /// Write out the plan as a dot language graph to specified path.
    pub fn to_dot_file<P: AsRef<std::path::Path>>(&self, path: &P) {
        println!("Writing plan dot: {:?}", path.as_ref());
        let mut writer =
            std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        writeln!(writer, "digraph plan {{").unwrap();

        for (i, node) in self.nodes.iter().enumerate() {
            writeln!(writer,
                     " n_{id} [label=\"n_{id}: [{s}, {e})\nt: {t:?}, layer: {l}\nsb: {aabb}\"];",
                     s = node.start_time,
                     e = node.end_time,
                     id = i,
                     l = node.layer,
                     t = node.n_type,
                     aabb = node.stencil_aabb).unwrap();
        }

        for (i, node) in self.nodes.iter().enumerate() {
            if let Some([n1, n2]) = node.sub_ops {
                writeln!(writer, " n_{} -> n_{};", i, n1).unwrap();
                writeln!(writer, " n_{} -> n_{};", i, n2).unwrap();
            }
        }
        writeln!(writer, "}}").unwrap();
    }
}
