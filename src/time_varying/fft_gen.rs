use crate::fft_solver::*;
use crate::time_varying::*;
use crate::util::*;
use std::collections::HashMap;

pub type FFTPairId = usize;

pub struct FFTGen<const GRID_DIMENSION: usize> {
    plans: Vec<FFTPlanPair>,
    key_map: HashMap<Coord<GRID_DIMENSION>, FFTPairId>,
    plan_type: PlanType,
}

impl<const GRID_DIMENSION: usize> FFTGen<GRID_DIMENSION> {
    pub fn new(plan_type: PlanType) -> Self {
        FFTGen {
            plans: Vec::new(),
            key_map: HashMap::new(),
            plan_type,
        }
    }

    pub fn get_op(
        &mut self,
        exclusive_bounds: Coord<GRID_DIMENSION>,
        threads: usize,
    ) -> FFTPairId {
        *self.key_map.entry(exclusive_bounds).or_insert_with(|| {
            let result = self.plans.len();
            self.plans.push(FFTPlanPair::create(
                exclusive_bounds,
                threads,
                self.plan_type,
            ));
            result
        })
    }

    pub fn finish(self) -> FFTStore {
        FFTStore::new(self.plans)
    }
}
