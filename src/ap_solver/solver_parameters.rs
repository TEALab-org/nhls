use crate::fft_solver::PlanType;
use crate::util::*;

/// Planner recurses by finding periodic solves.
/// These solves are configured with these parameters.
pub struct SolverParameters<const GRID_DIMENSION: usize> {
    pub steps: usize,
    pub plan_type: PlanType,
    pub cutoff: i32,
    pub ratio: f64,
    pub chunk_size: usize,
    pub threads: usize,
    pub aabb: AABB<GRID_DIMENSION>,
}
