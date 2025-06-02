use crate::domain::*;
use crate::stencil::TVStencil;
use crate::util::*;

// Used to direct solve frustrum regions.
pub struct TVDirectFrustrumSolver<
    'a,
    BC,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> where
    BC: BCCheck<GRID_DIMENSION>,
{
    pub bc: &'a BC,
    pub stencil: &'a StencilType,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
    pub chunk_size: usize,
}
