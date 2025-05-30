use crate::ap_solver::ap_periodic_ops::*;
use crate::ap_solver::index_types::*;
use crate::ap_solver::periodic_ops::*;
use crate::fft_solver::ConvolutionOperation;
use crate::fft_solver::PlanType;
use crate::stencil::*;
use crate::util::*;
use fftw::array::*;
use std::collections::HashMap;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct ConvolutionDescriptor<const GRID_DIMENSION: usize> {
    exclusive_bounds: Coord<GRID_DIMENSION>,
    steps: usize,
    threads: usize,
}

/// Used by APPlaner to create convolution operations,
/// and assign them IDs.
pub struct ApPeriodicOpsBuilder<
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> {
    stencil: &'a Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    operations: Vec<ConvolutionOperation>,
    real_buffer: AlignedVec<f64>,
    convolution_buffer: AlignedVec<c64>,
    plan_type: PlanType,
    key_map: HashMap<ConvolutionDescriptor<GRID_DIMENSION>, OpId>,
    chunk_size: usize,
}

impl<'a, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>
    ApPeriodicOpsBuilder<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
{
    pub fn new(
        max_aabb: &AABB<GRID_DIMENSION>,
        stencil: &'a Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        plan_type: PlanType,
        chunk_size: usize,
    ) -> Self {
        let max_real_size = max_aabb.buffer_size();
        let real_buffer = fftw::array::AlignedVec::new(max_real_size);
        let max_complex_size = max_aabb.complex_buffer_size();
        let convolution_buffer = fftw::array::AlignedVec::new(max_complex_size);

        ApPeriodicOpsBuilder {
            stencil,
            operations: Vec::new(),
            real_buffer,
            convolution_buffer,
            plan_type,
            key_map: HashMap::new(),
            chunk_size,
        }
    }

    pub fn get_op(
        &mut self,
        exclusive_bounds: Coord<GRID_DIMENSION>,
        steps: usize,
        threads: usize,
    ) -> OpId {
        let key = ConvolutionDescriptor {
            exclusive_bounds,
            steps,
            threads,
        };
        *self.key_map.entry(key).or_insert_with(|| {
            let result = self.operations.len();
            self.operations.push(ConvolutionOperation::create(
                self.stencil,
                &mut self.real_buffer,
                &mut self.convolution_buffer,
                &exclusive_bounds,
                steps,
                self.plan_type,
                self.chunk_size,
                threads,
            ));
            result
        })
    }

    pub fn op_count(&self) -> usize {
        self.operations.len()
    }

    pub fn finish(self) -> ApPeriodicOps {
        ApPeriodicOps::new(self.operations)
    }
}

impl<'a, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>
    PeriodicOpsBuilder<GRID_DIMENSION, ApPeriodicOps>
    for ApPeriodicOpsBuilder<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
{
    fn get_op_id(
        &mut self,
        descriptor: PeriodicOpDescriptor<GRID_DIMENSION>,
    ) -> OpId {
        self.get_op(
            descriptor.exclusive_bounds,
            descriptor.steps,
            descriptor.threads,
        )
    }

    fn finish(self) -> ApPeriodicOps {
        self.finish()
    }
}
