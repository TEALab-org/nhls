use crate::domain::*;
use crate::util::*;

pub trait SVDirectSolver<const GRID_DIMENSION: usize> {
    fn apply<'b>(
        &self,
        input_domain_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain_1: &mut SliceDomain<'b, GRID_DIMENSION>,
        input_domain_2: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain_2: &mut SliceDomain<'b, GRID_DIMENSION>,
        sloped_sides: &Bounds<GRID_DIMENSION>,
        steps: usize,
        global_time: usize,
        threads: usize,
    );
}
