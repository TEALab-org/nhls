use crate::par_slice;
use crate::solver::fft_plan::*;
use crate::stencil::*;
use std::collections::HashMap;

use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::*;

use crate::util::*;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct PeriodicPlanDescriptor<const GRID_DIMENSION: usize> {
    pub space_size: Bound<GRID_DIMENSION>,
    pub delta_t: usize,
}

impl<const GRID_DIMENSION: usize> PeriodicPlanDescriptor<GRID_DIMENSION> {
    fn new(space_size: Bound<GRID_DIMENSION>, delta_t: usize) -> Self {
        PeriodicPlanDescriptor {
            space_size,
            delta_t,
        }
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
    convolution_map: HashMap<PeriodicPlanDescriptor<GRID_DIMENSION>, AlignedVec<c32>>,
    fft_plan_library: FFTPlanLibrary<GRID_DIMENSION>,
    stencil: &'a StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
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

    pub fn apply(
        &mut self,
        bound: Bound<GRID_DIMENSION>,
        n: usize,
        input: &mut [f32],
        output: &mut [f32],
        complex_buffer: &mut [c32],
        chunk_size: usize,
    ) {
        let key = PeriodicPlanDescriptor::new(bound, n);
        // Can't do clippy fix on this line,
        // creating the new convolution requires mutable self borrow,
        // should probably break that out a bit so we can borrow self members separately.
        #[allow(clippy::map_entry)]
        if !self.convolution_map.contains_key(&key) {
            let new_convolution = self.new_convolution(&key, chunk_size);
            self.convolution_map.insert(key, new_convolution);
        }
        let convolution = self.convolution_map.get(&key).unwrap();
        let fft_plan = self.fft_plan_library.get_plan(bound);

        // fftw bindings expect slices of specific size
        let n_r = real_buffer_size(&bound);
        let n_c = complex_buffer_size(&bound);
        fft_plan
            .forward_plan
            .r2c(&mut input[0..n_r], &mut complex_buffer[0..n_c])
            .unwrap();
        par_slice::multiply_by(
            &mut complex_buffer[0..n_c],
            convolution.as_slice(),
            chunk_size,
        );
        fft_plan
            .backward_plan
            .c2r(&mut complex_buffer[0..n_c], &mut output[0..n_r])
            .unwrap();
        par_slice::div(&mut output[0..n_r], n_r as f32, chunk_size);
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
            let l = linear_index(&index, &descriptor.space_size);
            self.real_buffer[l] = self.stencil_weights[n_i];
        }

        // Calculate convolution of stencil
        let n_r = real_buffer_size(&descriptor.space_size);
        let n_c = complex_buffer_size(&descriptor.space_size);
        let fft_plan = self.fft_plan_library.get_plan(descriptor.space_size);
        fft_plan
            .forward_plan
            .r2c(
                &mut self.real_buffer[0..n_r],
                &mut self.convolution_buffer[0..n_c],
            )
            .unwrap();

        // clean up real buffer
        for n_i in 0..NEIGHBORHOOD_SIZE {
            let index = periodic_offset_index(&offsets[n_i], &descriptor.space_size);
            let l = linear_index(&index, &descriptor.space_size);
            self.real_buffer[l] = 0.0;
        }

        // Apply power calculation to convolution
        let complex_buffer_size = complex_buffer_size(&descriptor.space_size);
        let mut result_buffer = fftw::array::AlignedVec::new(complex_buffer_size);
        par_slice::power(
            descriptor.delta_t,
            &mut self.convolution_buffer[0..n_c],
            &mut result_buffer[0..n_c],
            chunk_size,
        );
        // overkill, could only set the values we know we touched,
        par_slice::set_value(
            &mut self.convolution_buffer[0..n_c],
            c32::zero(),
            chunk_size,
        );

        result_buffer
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use nalgebra::vector;

    fn test_unit_stencil<Operation, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>(
        stencil: &StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        bound: Bound<GRID_DIMENSION>,
        n: usize,
        plan_library: &mut PeriodicPlanLibrary<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    ) where
        Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
    {
        let chunk_size = 1;
        assert_eq!(stencil.apply(&[1.0; NEIGHBORHOOD_SIZE]), 1.0);
        let rbs = real_buffer_size(&bound);
        let cbs = complex_buffer_size(&bound);

        let mut input_x = fftw::array::AlignedVec::new(rbs + 4);
        for x in input_x.as_slice_mut() {
            *x = 1.0f32;
        }
        let input_copy = input_x.clone();
        let mut complex_buffer = fftw::array::AlignedVec::new(cbs + 10);
        let mut result_buffer = fftw::array::AlignedVec::new(rbs + 14);
        plan_library.apply(
            bound,
            n,
            &mut input_x,
            &mut result_buffer,
            &mut complex_buffer,
            chunk_size,
        );
        for x in &result_buffer[0..rbs] {
            assert_approx_eq!(f32, *x, 1.0);
        }
        assert_eq!(input_x.as_slice(), input_copy.as_slice());
    }

    #[test]
    fn test_1d_simple() {
        let stencil = Stencil::new([[0]], |args: &[f32; 1]| args[0]);
        let max_size = vector![100];
        let mut plan_library = PeriodicPlanLibrary::new(&max_size, &stencil);

        test_unit_stencil(&stencil, max_size, 10, &mut plan_library);
        test_unit_stencil(&stencil, vector![99], 20, &mut plan_library);
    }

    #[test]
    fn test_2d_simple() {
        let stencil = Stencil::new([[0, 0]], |args: &[f32; 1]| args[0]);
        let max_size = vector![50, 50];
        let mut plan_library = PeriodicPlanLibrary::new(&max_size, &stencil);
        test_unit_stencil(&stencil, max_size, 31, &mut plan_library);
    }

    #[test]
    fn test_2d_less_simple() {
        let stencil = Stencil::new(
            [[0, -1], [0, 1], [1, 0], [-1, 0], [0, 0]],
            |args: &[f32; 5]| {
                debug_assert_eq!(args.len(), 5);
                let mut r = 0.0;
                for a in args {
                    r += a / 5.0;
                }
                r
            },
        );
        let max_size = vector![50, 50];
        let mut plan_library = PeriodicPlanLibrary::new(&max_size, &stencil);
        test_unit_stencil(&stencil, max_size, 9, &mut plan_library);
    }

    #[test]
    fn test_1d_less_simple() {
        let stencil = Stencil::new([[-1], [1], [0]], |args: &[f32; 3]| {
            debug_assert_eq!(args.len(), 3);
            let mut r = 0.0;
            for a in args {
                r += a / 3.0
            }
            r
        });
        let max_size = vector![100];
        let mut plan_library = PeriodicPlanLibrary::new(&max_size, &stencil);
        test_unit_stencil(&stencil, max_size, 43, &mut plan_library);
    }

    #[test]
    fn test_3d() {
        let stencil = Stencil::new(
            [
                [0, 0, -2],
                [4, 5, 3],
                [0, -1, 0],
                [0, 1, 0],
                [1, 0, 0],
                [-1, 0, 4],
                [0, 0, 0],
            ],
            |args: &[f32; 7]| {
                let mut r = 0.0;
                for a in args {
                    r += a / 7.0;
                }
                r
            },
        );
        let max_size = vector![20, 20, 20];
        let mut plan_library = PeriodicPlanLibrary::new(&max_size, &stencil);
        test_unit_stencil(&stencil, max_size, 13, &mut plan_library);
        test_unit_stencil(&stencil, max_size, 14, &mut plan_library);
        test_unit_stencil(&stencil, vector![18, 18, 18], 14, &mut plan_library);
    }
}
