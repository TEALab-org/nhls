use crate::fft_solver::*;
use crate::util::*;
use fftw::plan::*;

pub struct FFTPlanPair {
    pub forward_plan: fftw::plan::Plan<f64, c64, fftw::plan::Plan64>,
    pub backward_plan: fftw::plan::Plan<c64, f64, fftw::plan::Plan64>,
}

impl FFTPlanPair {
    pub fn create<const DIMENSION: usize>(
        exclusive_bounds: Coord<DIMENSION>,
        threads: usize,
        plan_type: PlanType,
    ) -> Self {
        fftw::threading::plan_with_nthreads_f64(threads);

        let plan_size = exclusive_bounds.try_cast::<usize>().unwrap();

        let forward_plan = fftw::plan::R2CPlan64::aligned(
            plan_size.as_slice(),
            plan_type.to_fftw3_flag(),
        )
        .unwrap();

        let backward_plan = fftw::plan::C2RPlan64::aligned(
            plan_size.as_slice(),
            plan_type.to_fftw3_flag(),
        )
        .unwrap();

        FFTPlanPair {
            forward_plan,
            backward_plan,
        }
    }
}
