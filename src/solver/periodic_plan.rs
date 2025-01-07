use crate::domain::*;
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
    pub bound: AABB<GRID_DIMENSION>,
    pub steps: usize,
}

impl<const GRID_DIMENSION: usize> PeriodicPlanDescriptor<GRID_DIMENSION> {
    fn new(bound: AABB<GRID_DIMENSION>, steps: usize) -> Self {
        PeriodicPlanDescriptor { bound, steps }
    }
}

// We need storage for plans
pub struct PeriodicPlanLibrary<
    'a,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
{
    convolution_map:
        HashMap<PeriodicPlanDescriptor<GRID_DIMENSION>, AlignedVec<c64>>,
    fft_plan_library: FFTPlanLibrary<GRID_DIMENSION>,
    stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    real_buffer: AlignedVec<f64>,
    complex_buffer: AlignedVec<c64>,
    convolution_buffer: AlignedVec<c64>,
    stencil_weights: [f64; NEIGHBORHOOD_SIZE],
}

impl<
        'a,
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    > PeriodicPlanLibrary<'a, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
{
    pub fn new(
        max_bound: &AABB<GRID_DIMENSION>,
        stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        plan_type: PlanType,
    ) -> Self {
        // Zeroed out by construction
        let max_real_size = max_bound.buffer_size();
        let real_buffer = fftw::array::AlignedVec::new(max_real_size);

        let max_complex_size = max_bound.complex_buffer_size();
        let convolution_buffer = fftw::array::AlignedVec::new(max_complex_size);
        let complex_buffer = fftw::array::AlignedVec::new(max_complex_size);
        let stencil_weights = stencil.extract_weights();

        PeriodicPlanLibrary {
            convolution_map: HashMap::new(),
            fft_plan_library: FFTPlanLibrary::new(plan_type),
            stencil,
            real_buffer,
            convolution_buffer,
            complex_buffer,
            stencil_weights,
        }
    }

    pub fn apply<DomainType: DomainView<GRID_DIMENSION>>(
        &mut self,
        input: &mut DomainType,
        output: &mut DomainType,
        steps: usize,
        chunk_size: usize,
    ) {
        debug_assert_eq!(input.aabb(), output.aabb());
        let key = PeriodicPlanDescriptor::new(*input.aabb(), steps);
        // Can't do clippy fix on this line,
        // creating the new convolution requires mutable self borrow,
        // should probably break that out a bit so we can borrow self members separately.
        #[allow(clippy::map_entry)]
        if !self.convolution_map.contains_key(&key) {
            let new_convolution = self.new_convolution(&key, chunk_size);
            self.convolution_map.insert(key, new_convolution);
        }
        let convolution = self.convolution_map.get(&key).unwrap();
        let fft_plan = self.fft_plan_library.get_plan(*input.aabb());

        // fftw bindings expect slices of specific size
        let n_r = input.aabb().buffer_size();
        let n_c = input.aabb().complex_buffer_size();
        fft_plan
            .forward_plan
            .r2c(input.buffer_mut(), &mut self.complex_buffer[0..n_c])
            .unwrap();
        par_slice::multiply_by(
            &mut self.complex_buffer[0..n_c],
            convolution.as_slice(),
            chunk_size,
        );
        fft_plan
            .backward_plan
            .c2r(&mut self.complex_buffer[0..n_c], output.buffer_mut())
            .unwrap();
        par_slice::div(output.buffer_mut(), n_r as f64, chunk_size);
    }

    fn new_convolution(
        &mut self,
        descriptor: &PeriodicPlanDescriptor<GRID_DIMENSION>,
        chunk_size: usize,
    ) -> AlignedVec<c64> {
        // Place offsets in real buffer
        let offsets = self.stencil.offsets();
        for n_i in 0..NEIGHBORHOOD_SIZE {
            let rn_i: Coord<GRID_DIMENSION> = offsets[n_i] * -1;
            let index = &descriptor.bound.periodic_coord(&rn_i);
            let l = descriptor.bound.coord_to_linear(index);
            self.real_buffer[l] = self.stencil_weights[n_i];
        }

        // Calculate convolution of stencil
        let n_r = descriptor.bound.buffer_size();
        let n_c = descriptor.bound.complex_buffer_size();
        let fft_plan = self.fft_plan_library.get_plan(descriptor.bound);
        fft_plan
            .forward_plan
            .r2c(
                &mut self.real_buffer[0..n_r],
                &mut self.convolution_buffer[0..n_c],
            )
            .unwrap();

        // clean up real buffer
        for n_i in 0..NEIGHBORHOOD_SIZE {
            let rn_i: Coord<GRID_DIMENSION> = offsets[n_i] * -1;
            let index = &descriptor.bound.periodic_coord(&rn_i);
            let l = descriptor.bound.coord_to_linear(index);
            self.real_buffer[l] = 0.0;
        }

        // Apply power calculation to convolution
        let mut result_buffer = fftw::array::AlignedVec::new(n_c);
        par_slice::power(
            descriptor.steps,
            &mut self.convolution_buffer[0..n_c],
            &mut result_buffer[0..n_c],
            chunk_size,
        );
        // overkill, could only set the values we know we touched,
        par_slice::set_value(
            &mut self.convolution_buffer[0..n_c],
            c64::zero(),
            chunk_size,
        );

        result_buffer
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use nalgebra::matrix;

    fn test_unit_stencil<
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    >(
        stencil: &StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        bound: AABB<GRID_DIMENSION>,
        steps: usize,
        plan_library: &mut PeriodicPlanLibrary<
            Operation,
            GRID_DIMENSION,
            NEIGHBORHOOD_SIZE,
        >,
    ) where
        Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    {
        let chunk_size = 3;
        assert_approx_eq!(f64, stencil.apply(&[1.0; NEIGHBORHOOD_SIZE]), 1.0);

        let mut input_domain = OwnedDomain::new(bound);
        let mut output_domain = OwnedDomain::new(bound);

        input_domain.par_set_values(|_| 1.0, chunk_size);

        plan_library.apply(
            &mut input_domain,
            &mut output_domain,
            steps,
            chunk_size,
        );
        for x in output_domain.buffer() {
            assert_approx_eq!(f64, *x, 1.0);
        }
    }

    #[test]
    fn test_1d_simple() {
        let stencil = Stencil::new([[0]], |args: &[f64; 1]| args[0]);
        let max_size = AABB::new(matrix![0, 99]);
        let mut plan_library =
            PeriodicPlanLibrary::new(&max_size, &stencil, PlanType::Estimate);

        test_unit_stencil(&stencil, max_size, 10, &mut plan_library);
        test_unit_stencil(
            &stencil,
            AABB::new(matrix![0, 98]),
            20,
            &mut plan_library,
        );
    }

    #[test]
    fn test_2d_simple() {
        let stencil = Stencil::new([[0, 0]], |args: &[f64; 1]| args[0]);
        let bound = AABB::new(matrix![0, 49; 0, 49]);
        let mut plan_library =
            PeriodicPlanLibrary::new(&bound, &stencil, PlanType::Estimate);
        test_unit_stencil(&stencil, bound, 31, &mut plan_library);
    }

    #[test]
    fn test_2d_less_simple() {
        let stencil = Stencil::new(
            [[0, -1], [0, 1], [1, 0], [-1, 0], [0, 0]],
            |args: &[f64; 5]| {
                debug_assert_eq!(args.len(), 5);
                let mut r = 0.0;
                for a in args {
                    r += a / 5.0;
                }
                r
            },
        );
        let bound = AABB::new(matrix![0, 49; 0, 49]);
        let mut plan_library =
            PeriodicPlanLibrary::new(&bound, &stencil, PlanType::Estimate);
        test_unit_stencil(&stencil, bound, 9, &mut plan_library);
    }

    #[test]
    fn test_1d_less_simple() {
        let stencil = Stencil::new([[-1], [1], [0]], |args: &[f64; 3]| {
            debug_assert_eq!(args.len(), 3);
            let mut r = 0.0;
            for a in args {
                r += a / 3.0
            }
            r
        });
        let bound = AABB::new(matrix![0, 99]);
        let mut plan_library =
            PeriodicPlanLibrary::new(&bound, &stencil, PlanType::Estimate);
        test_unit_stencil(&stencil, bound, 43, &mut plan_library);
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
            |args: &[f64; 7]| {
                let mut r = 0.0;
                for a in args {
                    r += a / 7.0;
                }
                r
            },
        );
        let bound = AABB::new(matrix![0, 19; 0, 19; 0, 19]);
        let mut plan_library =
            PeriodicPlanLibrary::new(&bound, &stencil, PlanType::Estimate);
        test_unit_stencil(&stencil, bound, 13, &mut plan_library);
        test_unit_stencil(&stencil, bound, 14, &mut plan_library);
        test_unit_stencil(&stencil, bound, 5, &mut plan_library);
        test_unit_stencil(
            &stencil,
            AABB::new(matrix![0, 14; 0, 14; 0, 14]),
            5,
            &mut plan_library,
        );
    }
}
