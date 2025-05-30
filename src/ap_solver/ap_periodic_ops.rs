use crate::ap_solver::index_types::*;
use crate::ap_solver::periodic_ops::*;
use crate::domain::*;
use crate::fft_solver::ConvolutionOperation;
use crate::util::*;

/// This stores the convolution operations in
/// an APSolver instance.
pub struct ApPeriodicOps {
    operations: Vec<ConvolutionOperation>,
}

impl ApPeriodicOps {
    pub fn new(operations: Vec<ConvolutionOperation>) -> Self {
        ApPeriodicOps { operations }
    }

    pub fn get(&self, op: OpId) -> &ConvolutionOperation {
        &self.operations[op]
    }
}

impl<const GRID_DIMENSION: usize> PeriodicOps<GRID_DIMENSION>
    for ApPeriodicOps
{
    fn build_ops(&mut self, _global_time: usize) {}

    fn apply_operation<'a>(
        &'a self,
        op_id: OpId,
        input: &mut SliceDomain<'a, GRID_DIMENSION>,
        output: &mut SliceDomain<'a, GRID_DIMENSION>,
        complex_buffer: &mut [c64],
        _global_time: usize,
        chunk_size: usize,
    ) {
        self.get(op_id)
            .apply(input, output, complex_buffer, chunk_size);
    }
}
