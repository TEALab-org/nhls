use crate::domain::*;
use crate::util::*;
use fftw::plan::*;

use crate::par_slice;
use crate::solver::fft_plan::*;
use crate::stencil::*;
use fftw::array::*;
use fftw::types::*;
use std::collections::HashMap;

pub type PlanId = usize;
pub type NodeId = usize;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ConvolutionDescriptor<const GRID_DIMENSION: usize> {
    pub exclusive_bounds: Coord<GRID_DIMENSION>,
    pub steps: usize,
}

pub struct ConvolutionGenerator<
    'a,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
{
    stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    operations: Vec<ConvolutionOperation>,
    real_buffer: AlignedVec<f64>,
    convolution_buffer: AlignedVec<c64>,
    plan_type: PlanType,
    key_map: HashMap<ConvolutionDescriptor<GRID_DIMENSION>, OpId>,
    stencil_weights: [f64; NEIGHBORHOOD_SIZE],
    chunk_size: usize,
}

impl<
        'a,
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    > ConvolutionGenerator<'a, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
{
    pub fn new(
        max_aabb: &AABB<GRID_DIMENSION>,
        stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        plan_type: PlanType,
        chunk_size: usize,
    ) -> Self {
        let max_real_size = max_aabb.buffer_size();
        let real_buffer = fftw::array::AlignedVec::new(max_real_size);
        let max_complex_size = max_aabb.complex_buffer_size();
        let convolution_buffer = fftw::array::AlignedVec::new(max_complex_size);
        let stencil_weights = stencil.extract_weights();

        ConvolutionGenerator {
            stencil,
            operations: Vec::new(),
            real_buffer,
            convolution_buffer,
            plan_type,
            key_map: HashMap::new(),
            stencil_weights,
            chunk_size,
        }
    }

    pub fn create_convolution_op(
        &mut self,
        aabb: &AABB<GRID_DIMENSION>,
        steps: usize,
    ) -> ConvolutionOperation {
        let size = aabb.exclusive_bounds();
        let plan_size = size.try_cast::<usize>().unwrap();
        let forward_plan = fftw::plan::R2CPlan64::aligned(
            plan_size.as_slice(),
            self.plan_type.to_fftw3_flag(),
        )
        .unwrap();
        let backward_plan = fftw::plan::C2RPlan64::aligned(
            plan_size.as_slice(),
            self.plan_type.to_fftw3_flag(),
        )
        .unwrap();

        // Place offsets in real buffer
        let offsets = self.stencil.offsets();
        for n_i in 0..NEIGHBORHOOD_SIZE {
            let rn_i: Coord<GRID_DIMENSION> = offsets[n_i] * -1;
            let index = &aabb.periodic_coord(&rn_i);
            let l = aabb.coord_to_linear(index);
            self.real_buffer[l] = self.stencil_weights[n_i];
        }

        // Calculate convolution of stencil
        let n_r = aabb.buffer_size();
        let n_c = aabb.complex_buffer_size();
        forward_plan
            .r2c(
                &mut self.real_buffer[0..n_r],
                &mut self.convolution_buffer[0..n_c],
            )
            .unwrap();

        // clean up real buffer
        for n_i in 0..NEIGHBORHOOD_SIZE {
            let rn_i: Coord<GRID_DIMENSION> = offsets[n_i] * -1;
            let index = &aabb.periodic_coord(&rn_i);
            let l = aabb.coord_to_linear(index);
            self.real_buffer[l] = 0.0;
        }

        // Apply power calculation to convolution
        let mut result_buffer = fftw::array::AlignedVec::new(n_c);
        par_slice::power(
            steps,
            &mut self.convolution_buffer[0..n_c],
            &mut result_buffer[0..n_c],
            self.chunk_size,
        );

        par_slice::set_value(
            &mut self.convolution_buffer[0..n_c],
            c64::zero(),
            self.chunk_size,
        );

        ConvolutionOperation {
            forward_plan,
            backward_plan,
            convolution: result_buffer,
        }
    }

    pub fn get_op(bounds: AABB<GRID_DIMENSION>, steps: usize) -> OpId {
        self.key_map.entry

        if !self.convolution_map.contains_key(&key) {
            let new_convolution = self.new_convolution(&key, chunk_size);
            self.convolution_map.insert(key, new_convolution);
        }

    }
}

/*
pub struct PeriodicPlan {
    // Convolution buffer

    // Forward plan

    // backward plan
}

// Memory can be shared between main and gap step.
// Main step is bigger
// Remeber
// We solve FrustrumFFT
pub struct FFTNode<const GRID_DIMENSION: usize> {
    pub aabb: AABB<GRID_DIMENSION>,
    pub plan: FFTPlan,
    pub steps: usize,
}

pub struct PlanNode {
    pub plan: PlanId,
    // Plan (convolution + fftw3 plan)

    // Enum maybe?
}

pub struct APPlan<const GRID_DIMENSION: usize> {
    // Root plan (convolution + fftw3 plans)
    pub root_plan: PlanId,

    // Root complex buffer
    // Can be split for first two sub-domains
    pub c_buffers: Vec<AlignedVec<c64>>,

    // Domain input / output buffers

    // Each domain gets a recursive thing?
    pub frustrum_roots: [[NodeId; 2]; GRID_DIMENSION],
    pub frustrum_nodes: Vec<FrustrumAPPlan>,
    pub plans: Vec<PeriodicPlan>,
}

impl APPlan<const GRID_DIMENSION: usize> {
    pub fn new(aabb: &AABB<GRID_DIMENSION>, steps: usize) -> Self {
        let root_plan = 0;
        let root_complex_buffer_size = aabb.compex_buffer_size();
        let c_buffers = vec![AlignedVec::new(root_complex_buffer_size)];
        APPlan {
            root_plan,
            c_buffers,
            frustrum_roots: [[0; 2]; GRID_DIMENSION],
            frustrum_nodes: Vec::new(),
            plans:
        }
    }
}
*/
