use crate::decomposition::*;
use crate::domain::*;
use crate::solver::*;
use crate::stencil::*;
use crate::util::*;
use fftw::array::*;

pub struct APSolver<
    'a,
    BC,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    bc: &'a BC,
    stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    params: FFTSolveParams<GRID_DIMENSION>,
    periodic_lib:
        PeriodicPlanLibrary<'a, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    chunk_size: usize,
    slopes: Bounds<GRID_DIMENSION>,
}

impl<
        'a,
        BC,
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    > APSolver<'a, BC, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    pub fn new(
        bc: &'a BC,
        stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        cutoff: i32,
        ratio: f64,
        max_bound: &AABB<GRID_DIMENSION>,
        chunk_size: usize,
    ) -> Self {
        let params = FFTSolveParams {
            slopes: stencil.slopes(),
            cutoff,
            ratio,
        };

        let periodic_lib = PeriodicPlanLibrary::new(max_bound, stencil);

        let slopes = stencil.slopes();

        APSolver {
            bc,
            stencil,
            params,
            periodic_lib,
            chunk_size,
            slopes,
        }
    }

    pub fn rec_solve(
        &mut self,
        input: &mut Domain<'a, GRID_DIMENSION>,
        output: &mut Domain<'a, GRID_DIMENSION>,
        steps: usize,
    ) {
        let maybe_fft_solve =
            try_fftsolve(input.aabb(), &self.params, Some(steps));
        if maybe_fft_solve.is_none() {
            box_apply(
                self.bc,
                self.stencil,
                input,
                output,
                steps,
                self.chunk_size,
            );
            return;
        }

        let fft_solve = maybe_fft_solve.unwrap();

        // Output domain now has FFT solve
        self.periodic_lib.apply(
            input,
            output,
            fft_solve.steps,
            self.chunk_size,
        );

        // For each degree make domain
        let sub_domain_bounds =
            input.aabb().decomposition(&fft_solve.solve_region);
        let sub_domain_sloped_sides = decomposition_slopes::<GRID_DIMENSION>();

        for d in 0..GRID_DIMENSION {
            for r in 0..2 {
                // get bounds
                let output_aabb = sub_domain_bounds[d][r];
                let sloped_sides = sub_domain_sloped_sides[d][r];
                let input_aabb = trapezoid_input_region(
                    steps,
                    &output_aabb,
                    &sloped_sides,
                    &self.slopes,
                );

                // Make sub domain
                let mut input_buffer =
                    AlignedVec::new(input_aabb.buffer_size());
                let mut output_buffer =
                    AlignedVec::new(input_aabb.buffer_size());
                let mut input_domain =
                    Domain::new(input_aabb, &mut input_buffer);
                let mut output_domain =
                    Domain::new(input_aabb, &mut output_buffer);

                // copy input
                input_domain.par_from_superset(input, self.chunk_size);

                trapezoid_apply(
                    self.bc,
                    self.stencil,
                    &mut input_domain,
                    &mut output_domain,
                    &sloped_sides,
                    &self.slopes,
                    steps,
                    self.chunk_size,
                );

                // copy output
                debug_assert_eq!(output_domain.aabb(), &output_aabb);
                output.par_set_subdomain(&output_domain, self.chunk_size);
            }
        }
    }
}
