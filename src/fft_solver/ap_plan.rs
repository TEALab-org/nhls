use crate::fft_solver::*;
use crate::util::*;
use std::io::prelude::*;
use std::ops::Range;

/// A periodic solve is defined by an input and output AABB.
/// Note that the output AABB includes both the output of the
/// periodic solve and the boundary solves.
/// The boundary solve nodes are assumed to be a contiguous range of
/// nodes.
#[derive(Debug)]
pub struct PeriodicSolveNode<const GRID_DIMENSION: usize> {
    /// Required input buffer
    pub input_aabb: AABB<GRID_DIMENSION>,

    /// Output buffer that keep, includes boundary solves
    pub output_aabb: AABB<GRID_DIMENSION>,

    /// Which convolution operation to use (see `ConvolutionStore`).
    pub convolution_id: OpId,

    /// How many steps does our convolution solve for?
    pub steps: usize,

    /// Contiguous range of boundary solve nodes
    pub boundary_nodes: Range<NodeId>,

    /// Is there a time cut following this solve
    pub time_cut: Option<NodeId>,
}

/// Direct solves have an input and output AABB,
/// steps and sloped sides.
/// Strictly speaking we don't need the output_aabb,
/// but its remains useful for debugging.
#[derive(Debug)]
pub struct DirectSolveNode<const GRID_DIMENSION: usize> {
    pub input_aabb: AABB<GRID_DIMENSION>,
    pub output_aabb: AABB<GRID_DIMENSION>,
    pub sloped_sides: Bounds<GRID_DIMENSION>,
    pub steps: usize,
}

/// Used for central periodic solve, can't appear in frustrums.
/// However, the largest central periodic solve we can find may
/// need to be repeated many times to achieve the desired number of steps.
/// Possible followed by a single periodic solve to get the remainder
/// of steps.
#[derive(Debug)]
pub struct RepeatNode {
    pub n: usize,
    pub node: NodeId,
    pub next: Option<NodeId>,
}

/// Use for root node in time-varying solvers
#[derive(Debug)]
pub struct RangeNode {
    pub range: Range<NodeId>,
}

/// These nodes form a tree.
#[derive(Debug)]
pub enum PlanNode<const GRID_DIMENSION: usize> {
    PeriodicSolve(PeriodicSolveNode<GRID_DIMENSION>),
    DirectSolve(DirectSolveNode<GRID_DIMENSION>),
    Repeat(RepeatNode),
    Range(RangeNode),
}

/// An `APPlan` describes an aperiodic solve over a fixed AABB
/// for fixed number of time steps.
/// The root node should always be the only repeat node in the tree.
pub struct APPlan<const GRID_DIMENSION: usize> {
    pub nodes: Vec<PlanNode<GRID_DIMENSION>>,
    pub root: NodeId,
}

impl<const GRID_DIMENSION: usize> APPlan<GRID_DIMENSION> {
    /// Retrieve a node
    pub fn get_node(&self, node: NodeId) -> &PlanNode<GRID_DIMENSION> {
        &self.nodes[node]
    }

    /// Retrieve periodic node at node_id, will panic if type is incorrect.
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

    /// Retrieve direct node at node_id, will panic if type is incorrect.
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

    /// Retrieve repeat node at node_id, will panic if type is incorrect.
    #[track_caller]
    pub fn unwrap_repeat_node(&self, node_id: NodeId) -> &RepeatNode {
        if let PlanNode::Repeat(repeat_node) = self.get_node(node_id) {
            repeat_node
        } else {
            panic!("ERROR: Not a repeat node, {}", node_id);
        }
    }

    /// Number of nodes in the plan
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check whether the plan is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Write out the plan as a dot language graph to specified path.
    pub fn to_dot_file<P: AsRef<std::path::Path>>(&self, path: &P) {
        println!("Writing plan dot: {:?}", path.as_ref());
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
                PlanNode::Range(_) => {
                    writeln!(
                        writer,
                        " n_{id} [label=\"n_{id}: RANGE\"];",
                        id = i,
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
                PlanNode::DirectSolve(_) => {}
                PlanNode::Repeat(r) => {
                    writeln!(writer, " n_{} -> n_{} [color=green];", i, r.node)
                        .unwrap();
                    if let Some(r2) = r.next {
                        writeln!(writer, " n_{} -> n_{} [color=black];", i, r2)
                            .unwrap();
                    }
                }
                PlanNode::Range(r) => {
                    for r in r.range.clone() {
                        writeln!(writer, " n_{} -> n_{} [color=green];", i, r)
                            .unwrap();
                    }
                }
            }
        }

        writeln!(writer, "}}").unwrap();
    }
}
