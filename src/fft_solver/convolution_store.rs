use crate::fft_solver::*;

/// This stores the convolution operations in
/// an APSolver instance.
pub struct ConvolutionStore {
    operations: Vec<ConvolutionOperation>,
}

impl ConvolutionStore {
    pub fn new(operations: Vec<ConvolutionOperation>) -> Self {
        ConvolutionStore { operations }
    }

    pub fn get(&self, op: OpId) -> &ConvolutionOperation {
        &self.operations[op]
    }
}
