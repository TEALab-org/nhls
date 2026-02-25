/*
use crate::util::*;
use crate::stencil::*;
use crate::sparse::dynamic_boundary::*;
pub struct Boundary<const GRID_DIMENSION: usize> {
    dynamic_boundary_in: BoundarySet<GRID_DIMENSION>,
    dynamic_boundary_out: BoundarySet<GRID_DIMENSION>,
    fixed_boundary_in: BoundarySet<GRID_DIMENSION>,
    fixed_boundary_out: BoundarySet<GRID_DIMENSION>,
}

impl <const GRID_DIMENSION: usize> Boundary<GRID_DIMENSION> {
    pub fn new(dbi: BoundarySet<GRID_DIMENSION>, dbo: BoundarySet<GRID_DIMENSION>, fbi: BoundarySet<GRID_DIMENSION>, fbo: BoundarySet<GRID_DIMENSION>) -> Self {
        Self {
            dynamic_boundary_in: dbi,
            dynamic_boundary_out: dbo,
            fixed_boundary_in: fbi,
            fixed_boundary_out: fbo,
        }
    }

    pub fn dilate_in<const NEIGHBORHOOD_SIZE: usize>(&self, stencil: &Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>) -> Self {
        let mut new_dbi = BoundarySet::empty();
        let mut new_dbo = BoundarySet::empty();
        let mut new_fbi = BoundarySet::empty();
        let mut new_fbo = BoundarySet::empty();

        // Create new dynamic in / out
        for in_coord in self.dynamic_boundary_in.coord_iter() {
            for offset in stencil.offsets() {
                let n_coord = in_coord + offset;
                //if self.dynamic_boundary_out.contains
            }
        }


        // Trim fbi, fbo,

        Self {
            dynamic_boundary_in: new_dbi,
            dynamic_boundary_out: new_dbo,
            fixed_boundary_in: new_fbi,
            fixed_boundary_out: new_fbo,
        }
    }
}
*/
