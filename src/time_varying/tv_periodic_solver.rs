use crate::domain::*;
use crate::time_varying::*;
use crate::util::*;

pub struct FFTPlanPair {
    pub forward_plan: fftw::plan::Plan<f64, c64, fftw::plan::Plan64>,
    pub backward_plan: fftw::plan::Plan<c64, f64, fftw::plan::Plan64>,
}

pub struct TVPeriodicSolver<
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> {
    stencil: &'a StencilType,
    stencil_slopes: Bounds<GRID_DIMENSION>,
    steps: usize,
    tree_plans: Vec<FFTPlanPair>,
    stencil_domains: Vec<Vec<OwnedDomain<GRID_DIMENSION>>>,
    threads: usize,
    tree_levels: usize,
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    > TVPeriodicSolver<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
{
    pub fn new(stencil: &'a StencilType, steps: usize, threads: usize) -> Self {
        let stencil_slopes = stencil.slopes();

        TVPeriodicSolver {
            stencil,
            stencil_slopes,
            steps,
            tree_plans: Vec::new(),
            stencil_domains: Vec::new(),
            threads,
            tree_levels: 0,
        }
    }

    pub fn apply<DomainType: DomainView<GRID_DIMENSION>>(
        &self,
        input: &mut DomainType,
        output: &mut DomainType,
        mut global_time: usize,
    ) {
        // construct tree
        // until only final convolution remains
        // reduce every 2,
        // every four
        // until ln self.stepsA
        //

        for level in 0..self.tree_levels {}
    }
}
