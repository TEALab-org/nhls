use crate::domain::*;
use crate::fft_solver::*;
use crate::stencil::*;
use crate::util::*;
use fftw::array::*;

pub struct PeriodicSolver {
    operation: ConvolutionOperation,
    complex_buffer: AlignedVec<c64>,
    chunk_size: usize,
}

impl PeriodicSolver {
    pub fn create<
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    >(
        stencil: &StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        real_buffer: &mut [f64],
        aabb: &AABB<GRID_DIMENSION>,
        steps: usize,
        plan_type: PlanType,
        chunk_size: usize,
    ) -> Self
    where
        Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    {
        let stencil_weights = stencil.extract_weights();
        let mut complex_buffer = AlignedVec::new(aabb.complex_buffer_size());
        let operation = ConvolutionOperation::create(
            stencil,
            &stencil_weights,
            real_buffer,
            &mut complex_buffer,
            aabb,
            steps,
            plan_type,
            chunk_size,
        );

        PeriodicSolver {
            operation,
            complex_buffer,
            chunk_size,
        }
    }

    pub fn apply<
        const GRID_DIMENSION: usize,
        DomainType: DomainView<GRID_DIMENSION>,
    >(
        &mut self,
        input: &mut DomainType,
        output: &mut DomainType,
    ) {
        self.operation.apply(
            input,
            output,
            &mut self.complex_buffer,
            self.chunk_size,
        );
    }
}
