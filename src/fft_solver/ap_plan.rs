use crate::fft_solver::*;
use crate::util::*;
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
    pub remainder: Range<NodeId>,

    /// Should we swap input / output and run this?
    pub next: Option<NodeId>,
}

#[derive(Debug)]
pub struct DirectSolveNode<const GRID_DIMENSION: usize> {
    pub input_aabb: AABB<GRID_DIMENSION>,
    pub output_aabb: AABB<GRID_DIMENSION>,
    pub steps: usize,
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
}
