use crate::fft_solver::PlanType;
use crate::util::*;

/// Solver generation is configurable.
/// These are all the parameters.
pub struct SolverParameters<const GRID_DIMENSION: usize> {
    /// Number of steps for one `apply` operation.
    pub steps: usize,

    /// Optimization level for FFTW3 plans.
    pub plan_type: PlanType,

    /// Per axis cutoff for applying periodic solves.
    pub cutoff: i32,

    /// Approximate output size for periodic solves.
    pub ratio: f64,

    /// Some multithreaded operations on vectors are chunked with this size
    pub chunk_size: usize,

    /// How many threads are available for `apply` operatons.
    pub threads: usize,

    /// Domain bounds
    pub aabb: AABB<GRID_DIMENSION>,

    /// Minimum number of tasks a plan node can use
    pub task_min: usize,

    /// Assume total tasks available relative to threads
    pub task_mult: f64,
}

impl<const GRID_DIMENSION: usize> std::default::Default
    for SolverParameters<GRID_DIMENSION>
{
    fn default() -> Self {
        SolverParameters {
            steps: 100,
            plan_type: PlanType::Estimate,
            cutoff: 100,
            ratio: 0.5,
            chunk_size: 1000,
            threads: 1,
            aabb: AABB::new(Bounds::zeros()),
            task_min: 1,
            task_mult: 1.0,
        }
    }
}
