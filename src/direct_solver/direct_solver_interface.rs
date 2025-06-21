use crate::domain::*;
use crate::util::*;

/// This interface is specifically for direct solvers used
/// in the context of our aperiodic algorithm.
/// Note that the trapezoidal region implied by
/// * Input domain aabb
/// * sloped sides
/// * stencil slopes
/// means that some direct solvers may choose to shrink the domain
/// each time step, though this is required.
pub trait DirectSolverInterface<const GRID_DIMENSION: usize>:
    Send + Sync
{
    fn apply<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        sloped_sides: &Bounds<GRID_DIMENSION>,
        steps: usize,
        global_time: usize,
        threads: usize,
    );
}
