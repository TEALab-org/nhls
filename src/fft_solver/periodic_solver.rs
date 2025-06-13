use crate::domain::*;
use crate::fft_solver::*;
use crate::stencil::*;
use crate::util::*;
use fftw::array::*;
use indexing::complex_buffer_size;

pub struct PeriodicSolver {
    operation: ConvolutionOperation,
    complex_buffer: AlignedVec<c64>,
    chunk_size: usize,
}

impl PeriodicSolver {
    pub fn create<
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    >(
        stencil: &Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        real_buffer: &mut [f64],
        aabb: &AABB<GRID_DIMENSION>,
        steps: usize,
        plan_type: PlanType,
        chunk_size: usize,
        threads: usize,
    ) -> Self {
        let exclusive_bounds = aabb.exclusive_bounds();
        let mut complex_buffer =
            AlignedVec::new(complex_buffer_size(&exclusive_bounds));
        let operation = ConvolutionOperation::create(
            stencil,
            real_buffer,
            &mut complex_buffer,
            &exclusive_bounds,
            steps,
            plan_type,
            chunk_size,
            threads,
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

#[cfg(test)]
mod unit_tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use nalgebra::matrix;

    fn test_unit_stencil<
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    >(
        stencil: &Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        aabb: AABB<GRID_DIMENSION>,
        steps: usize,
    ) {
        let threads = 1;
        let chunk_size = 3;
        let plan_type = PlanType::Estimate;
        assert_approx_eq!(f64, stencil.apply(&Values::from_element(1.0)), 1.0);

        let mut input_domain = OwnedDomain::new(aabb);
        let mut output_domain = OwnedDomain::new(aabb);

        input_domain.par_set_values(|_| 1.0, chunk_size);

        let mut solver = PeriodicSolver::create(
            stencil,
            output_domain.buffer_mut(),
            &aabb,
            steps,
            plan_type,
            chunk_size,
            threads,
        );
        solver.apply(&mut input_domain, &mut output_domain);
        for x in output_domain.buffer() {
            assert_approx_eq!(f64, *x, 1.0);
        }
    }

    #[test]
    fn test_1d_simple() {
        let stencil = Stencil::new([[0]], |args: &[f64; 1]| args[0]);
        let max_size = AABB::new(matrix![0, 99]);

        test_unit_stencil(&stencil, max_size, 10);
        test_unit_stencil(&stencil, AABB::new(matrix![0, 98]), 20);
    }

    #[test]
    fn test_2d_simple() {
        let stencil = Stencil::new([[0, 0]], |args: &[f64; 1]| args[0]);
        let bound = AABB::new(matrix![0, 49; 0, 49]);
        test_unit_stencil(&stencil, bound, 31);
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
        test_unit_stencil(&stencil, bound, 9);
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
        test_unit_stencil(&stencil, bound, 43);
    }

    #[test]
    fn test_3d() {
        let stencil = Stencil::<3, 7>::new(
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
        test_unit_stencil(&stencil, bound, 13);
        test_unit_stencil(&stencil, bound, 14);
        test_unit_stencil(&stencil, bound, 5);
        test_unit_stencil(&stencil, AABB::new(matrix![0, 14; 0, 14; 0, 14]), 5);
    }

    #[test]
    fn shifter() {
        let chunk_size = 1;
        let plan_type = PlanType::Estimate;
        let stencil = Stencil::new([[-1]], |args: &[f64; 1]| args[0]);
        let aabb = AABB::new(matrix![0, 9]);

        let mut input_domain = OwnedDomain::new(aabb);
        let mut output_domain = OwnedDomain::new(aabb);
        input_domain
            .par_set_values(|coord: Coord<1>| coord[0] as f64, chunk_size);

        let n = 1;
        let mut solver = PeriodicSolver::create(
            &stencil,
            output_domain.buffer_mut(),
            &aabb,
            n,
            plan_type,
            chunk_size,
            1,
        );
        solver.apply(&mut input_domain, &mut output_domain);
        for i in 0..10 {
            assert_approx_eq!(
                f64,
                output_domain.buffer()[(i + n) % 10],
                i as f64
            );
        }
    }
}
