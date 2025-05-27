use crate::domain::*;
use crate::util::*;

pub trait DirectSolver<const GRID_DIMENSION: usize>: Send + Sync {
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
