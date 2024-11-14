use crate::par_slice;
use crate::solver::fft_plan::*;
use crate::solver::*;
use crate::stencil::*;
use nalgebra::*;
use std::collections::HashMap;

use crate::par_slice;

use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::*;

use crate::util::*;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PeriodicPlanDescriptor<const GRID_DIMENSION: usize> {
    pub space_size: Bound<GRID_DIMENSION>,
    pub delta_t: usize,
}

impl<const GRID_DIMENSION: usize> PeriodicPlanDescriptor<GRID_DIMENSION> {
    pub fn new(space_size: Bound<GRID_DIMENSION>, delta_t: usize) -> Self {
        PeriodicPlanDescriptor {
            space_size,
            delta_t,
        }
    }
}

pub struct PeriodicPlan<'a> {
    pub fft_plan: &'a mut FFTPlan,
    pub convolution: &'a AlignedVec<c32>,
}

impl<'a> PeriodicPlan<'a> {
    pub fn apply(
        &mut self,
        mut input: AlignedVec<f32>,
        output: &mut AlignedVec<f32>,
        complex_buffer: &mut AlignedVec<c32>,
        chunk_size: usize,
    ) -> AlignedVec<f32> {
        self.fft_plan
            .forward_plan
            .r2c(&mut input, complex_buffer)
            .unwrap();
        par_slice::multiply_by(complex_buffer, self.convolution.as_slice(), chunk_size);
        self.fft_plan
            .backward_plan
            .c2r(complex_buffer, output)
            .unwrap();
        input
    }
}

// We need storage for plans
pub struct PeriodicPlanLibrary<
    'a,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
{
    pub convolution_map: HashMap<PeriodicPlanDescriptor<GRID_DIMENSION>, AlignedVec<c32>>,
    pub fft_plan_library: FFTPlanLibrary<GRID_DIMENSION>,
    pub stencil: &'a StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    real_buffer: AlignedVec<f32>,
    convolution_buffer: AlignedVec<c32>,
    stencil_weights: [f32; NEIGHBORHOOD_SIZE],
}

impl<'a, Operation, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>
    PeriodicPlanLibrary<'a, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
{
    pub fn new(
        max_size: &Bound<GRID_DIMENSION>,
        stencil: &'a StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    ) -> Self {
        // Zeroed out by construction
        let max_real_size = real_buffer_size(max_size);
        let real_buffer = fftw::array::AlignedVec::new(max_real_size);

        let max_complex_size = complex_buffer_size(max_size);
        let convolution_buffer = fftw::array::AlignedVec::new(max_complex_size);

        let stencil_weights = stencil.extract_weights();

        PeriodicPlanLibrary {
            convolution_map: HashMap::new(),
            fft_plan_library: FFTPlanLibrary::new(),
            stencil,
            real_buffer,
            convolution_buffer,
            stencil_weights,
        }
    }

    pub fn get_plan(
        &'a mut self,
        size: Bound<GRID_DIMENSION>,
        delta_t: usize,
        chunk_size: usize,
    ) -> PeriodicPlan<'a> {
        let key = PeriodicPlanDescriptor::new(size, delta_t);
        if !self.convolution_map.contains_key(&key) {
            let new_convolution = self.new_convolution(&key, chunk_size);
            self.convolution_map.insert(key, new_convolution);
        }
        let convolution = self.convolution_map.get(&key).unwrap();
        let fft_plan = self.fft_plan_library.get_plan(size);

        PeriodicPlan {
            convolution,
            fft_plan,
        }
    }

    fn new_convolution(
        &mut self,
        descriptor: &PeriodicPlanDescriptor<GRID_DIMENSION>,
        chunk_size: usize,
    ) -> AlignedVec<c32> {
        // Place offsets in real buffer
        let offsets = self.stencil.offsets();
        for n_i in 0..NEIGHBORHOOD_SIZE {
            let index = periodic_offset_index(&offsets[n_i], &descriptor.space_size);
            let l = linear_index(index, descriptor.space_size);
            self.real_buffer[l] = self.stencil_weights[n_i];
        }

        // Calculate convolution of stencil
        let fft_plan = self.fft_plan_library.get_plan(descriptor.space_size);
        fft_plan
            .forward_plan
            .r2c(&mut self.real_buffer, &mut self.convolution_buffer)
            .unwrap();

        // clean up real buffer
        for n_i in 0..NEIGHBORHOOD_SIZE {
            let index = periodic_offset_index(&offsets[n_i], &descriptor.space_size);
            let l = linear_index(index, descriptor.space_size);
            self.real_buffer[l] = 0.0;
        }

        // Apply power calculation to convolution
        let complex_buffer_size = complex_buffer_size(&descriptor.space_size);
        let mut result_buffer = fftw::array::AlignedVec::new(complex_buffer_size);
        par_slice::power(
            descriptor.delta_t,
            &mut self.convolution_buffer,
            &mut result_buffer,
            chunk_size,
        );
        // overkill, could only set the values we know we touched,
        par_slice::set_value(&mut self.convolution_buffer, c32::zero(), chunk_size);

        result_buffer
    }
}
