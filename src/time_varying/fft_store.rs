use crate::time_varying::*;

pub struct FFTStore {
    plans: Vec<FFTPlanPair>,
}

impl FFTStore {
    pub fn new(plans: Vec<FFTPlanPair>) -> Self {
        FFTStore { plans }
    }

    pub fn get(&self, op: FFTPairId) -> &FFTPlanPair {
        &self.plans[op]
    }
}
