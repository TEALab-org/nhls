use crate::fft_solver::*;
use crate::util::*;
use std::io::prelude::*;
use std::ops::Range;

#[derive(Debug)]
pub struct PeriodicSolveNode<const GRID_DIMENSION: usize> {
    pub input_aabb: AABB<GRID_DIMENSION>,
    pub output_aabb: AABB<GRID_DIMENSION>,
    pub convolution_id: OpId,
    pub steps: usize,

    /// calculate remaining output dections based on either
    ///  - AABB::decomposition for central solve
    ///  - APFrustrum::decomposition for frustrum solves
    pub boundary_nodes: Range<NodeId>,

    /// Should we swap input / output and run this?
    pub time_cut: Option<NodeId>,
}

#[derive(Debug)]
pub struct DirectSolveNode<const GRID_DIMENSION: usize> {
    pub input_aabb: AABB<GRID_DIMENSION>,
    pub output_aabb: AABB<GRID_DIMENSION>,
    pub sloped_sides: Bounds<GRID_DIMENSION>,
    pub steps: usize,
    pub out_of_bounds_cut: Option<NodeId>,
}

/// Used for central periodic solve, can't appear in frustrums
#[derive(Debug)]
pub struct RepeatNode {
    pub n: usize,
    pub node: NodeId,
    pub next: Option<NodeId>,
}

#[derive(Debug)]
pub enum PlanNode<const GRID_DIMENSION: usize> {
    PeriodicSolve(PeriodicSolveNode<GRID_DIMENSION>),
    DirectSolve(DirectSolveNode<GRID_DIMENSION>),
    Repeat(RepeatNode),
}

pub struct APPlan<const GRID_DIMENSION: usize> {
    pub nodes: Vec<PlanNode<GRID_DIMENSION>>,
    pub root: NodeId,
}

impl<const GRID_DIMENSION: usize> APPlan<GRID_DIMENSION> {
    pub fn get_node(&self, node: NodeId) -> &PlanNode<GRID_DIMENSION> {
        &self.nodes[node]
    }

    #[track_caller]
    pub fn unwrap_periodic_node(
        &self,
        node_id: NodeId,
    ) -> &PeriodicSolveNode<GRID_DIMENSION> {
        if let PlanNode::PeriodicSolve(periodic_node) = self.get_node(node_id) {
            periodic_node
        } else {
            panic!("ERROR: Not a periodic node, {}", node_id);
        }
    }

    #[track_caller]
    pub fn unwrap_direct_node(
        &self,
        node_id: NodeId,
    ) -> &DirectSolveNode<GRID_DIMENSION> {
        if let PlanNode::DirectSolve(direct_node) = self.get_node(node_id) {
            direct_node
        } else {
            panic!("ERROR: Not a direct node, {}", node_id);
        }
    }

    #[track_caller]
    pub fn unwrap_repeat_node(&self, node_id: NodeId) -> &RepeatNode {
        if let PlanNode::Repeat(repeat_node) = self.get_node(node_id) {
            repeat_node
        } else {
            panic!("ERROR: Not a repeat node, {}", node_id);
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn to_dot_file<P: AsRef<std::path::Path>>(&self, path: &P) {
        let mut writer =
            std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        writeln!(writer, "digraph plan {{").unwrap();

        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                PlanNode::PeriodicSolve(periodic_solve) => {
                    writeln!(
                        writer,
                        " n_{id} [label=\"n_{id}: PERIODIC\nsteps: {s}\nin: {in}\nout: {out}\nconv_id: {c_id}\"];",
                        id = i,
                        s = periodic_solve.steps,
                        in = periodic_solve.input_aabb,
                        out = periodic_solve.output_aabb,
                        c_id = periodic_solve.convolution_id,
                    )
                    .unwrap();
                }
                PlanNode::DirectSolve(direct_solve) => {
                    writeln!(
                        writer,
                        " n_{id} [label=\"n_{id}: DIRECT\nsteps: {s}\nin: {in}\nout: {out}\nsloped_sides: {slope:?}\"];",
                        id = i,
                        s = direct_solve.steps,
                        in = direct_solve.input_aabb,
                        out = direct_solve.output_aabb,
                        slope = direct_solve.sloped_sides,
                    )
                    .unwrap();
                }
                PlanNode::Repeat(repeat_node) => {
                    writeln!(
                        writer,
                        " n_{id} [label=\"n_{id}: REPEAT\nn: {n}\"];",
                        id = i,
                        n = repeat_node.n,
                    )
                    .unwrap();
                }
            }
        }

        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                PlanNode::PeriodicSolve(p) => {
                    for r in p.boundary_nodes.clone() {
                        writeln!(writer, " n_{} -> n_{} [color=blue];", i, r)
                            .unwrap();
                    }
                    if let Some(r) = p.time_cut {
                        writeln!(writer, " n_{} -> n_{} [color=red];", i, r)
                            .unwrap();
                    }
                }
                PlanNode::DirectSolve(p) => {
                    if let Some(r) = p.out_of_bounds_cut {
                        writeln!(
                            writer,
                            " n_{} -> n_{} [color = purple];",
                            i, r
                        )
                        .unwrap();
                    }
                }
                PlanNode::Repeat(r) => {
                    writeln!(writer, " n_{} -> n_{} [color=green];", i, r.node)
                        .unwrap();
                    if let Some(r2) = r.next {
                        writeln!(writer, " n_{} -> n_{} [color=black];", i, r2)
                            .unwrap();
                    }
                }
            }
        }

        writeln!(writer, "}}").unwrap();
    }
}
