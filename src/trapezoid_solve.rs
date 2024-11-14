/*
use crate::util::*;
use crate::stencil::*;
use nalgebra;

pub fn trapezoid_solve <
    NumType: NumTrait,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    Operation,
> (
    stencil: &Stencil<NumType, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    boundary_slopes: Slopes<GRID_DIMENSION>,
    end_min: Bound<GRID_DIMENSION>,
    end_max: Bound<GRID_DIMENSION>,
    t: i32,
) where Operation: StencilOperation<NumType, NEIGHBORHOOD_SIZE> {
    let stencil_slopes = stencil.slopes();
    let slopes = boundary_slopes.component_mul(stencil_slopes);
}

// Get radius

// compute slopes

// compute start min and start max
*/
