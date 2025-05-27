use crate::ap_solver::index_types::*;
use crate::domain::*;
use crate::util::*;

/// Describes a periodic solve,
/// including time-varying
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PeriodicOpDescriptor<const GRID_DIMENSION: usize> {
    pub step_min: usize,
    pub step_max: usize,
    pub steps: usize,
    pub exclusive_bounds: Coord<GRID_DIMENSION>,
    pub threads: usize,
}

pub trait PeriodicOpsBuilder<
    const GRID_DIMENSION: usize,
    SolverType: PeriodicOps<GRID_DIMENSION>,
>
{
    fn blank() -> Self;

    fn get_op_id(
        &mut self,
        descriptor: PeriodicOpDescriptor<GRID_DIMENSION>,
    ) -> OpId;

    fn finish(self) -> SolverType;
}

pub trait PeriodicOps<const GRID_DIMENSION: usize>: Send + Sync {
    fn blank() -> Self;

    fn build_ops(&mut self, global_time: usize);

    fn apply_operation(
        &self,
        op_id: OpId,
        input_domain: &SliceDomain<GRID_DIMENSION>,
        output_domain: &mut SliceDomain<GRID_DIMENSION>,
        // NOTE: TV will need twice the buffer, and we can split it as needed
        complex_buffer: &mut [c64],
        chunk_size: usize,
    );
}
