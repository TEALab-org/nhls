use crate::util::*;

pub trait Lookup<const GRID_DIMENSION: usize>: Sync {
    fn value(&self, coord: &Coord<GRID_DIMENSION>, input: &[f32]) -> f32;
}

pub struct IdentityLookup<const grid_dimension: usize> {
    bound: Coord<grid_dimension>
}

impl <const GRID_DIMENSION: usize> IdentityLookup<GRID_DIMENSION> {
    pub fn new(bound: Coord<GRID_DIMENSION>) -> Self {
        IdentityLookup {
            bound,
        }
    }
}

impl <const GRID_DIMENSION: usize> for IdentityLookup<GRID_DIMENSION> {
    fn value(&self, coord: &Coord<GRID_DIMENSION>, input: &[f32]) -> f32 {
        let linear_index = linear_index(coord, &self.bound);
        input[linear_index]
    }
}




