use fftw::plan::*;
use fftw::types::*;

use crate::util::*;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct FFTPlanDescriptor<const GRID_DIMENSION: usize> {
    pub bound: AABB<GRID_DIMENSION>,
}

impl<const GRID_DIMENSION: usize> FFTPlanDescriptor<GRID_DIMENSION> {
    pub fn new(bound: AABB<GRID_DIMENSION>) -> Self {
        FFTPlanDescriptor { bound }
    }
}

pub struct FFTPlan {
    pub forward_plan: fftw::plan::Plan<f32, c32, fftw::plan::Plan32>,
    pub backward_plan: fftw::plan::Plan<c32, f32, fftw::plan::Plan32>,
}

impl FFTPlan {
    pub fn new<const GRID_DIMENSION: usize>(bound: AABB<GRID_DIMENSION>) -> Self {
        let size = bound.exclusive_bounds();
        let plan_size = size.try_cast::<usize>().unwrap();
        let forward_plan =
            fftw::plan::R2CPlan32::aligned(plan_size.as_slice(), fftw::types::Flag::ESTIMATE)
                .unwrap();
        let backward_plan =
            fftw::plan::C2RPlan32::aligned(plan_size.as_slice(), fftw::types::Flag::ESTIMATE)
                .unwrap();
        FFTPlan {
            forward_plan,
            backward_plan,
        }
    }
}

// We need storage for plans
pub struct FFTPlanLibrary<const GRID_DIMENSION: usize> {
    pub plan_map: std::collections::HashMap<FFTPlanDescriptor<GRID_DIMENSION>, FFTPlan>,
}

impl<const GRID_DIMENSION: usize> FFTPlanLibrary<GRID_DIMENSION> {
    pub fn new() -> Self {
        FFTPlanLibrary {
            plan_map: std::collections::HashMap::new(),
        }
    }

    pub fn get_plan(&mut self, bound: AABB<GRID_DIMENSION>) -> &mut FFTPlan {
        let key = FFTPlanDescriptor::new(bound);
        self.plan_map.entry(key).or_insert(FFTPlan::new(bound))
    }
}
