use crate::domain::*;

/// All solvers should adhere implement this interface
pub trait SolverInterface<const GRID_DIMENSION: usize> {
    fn apply<'a>(
        &mut self,
        input_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        global_time: usize,
    );

    fn print_report(&self);

    fn to_dot_file<P: AsRef<std::path::Path>>(&self, path: &P);
}
