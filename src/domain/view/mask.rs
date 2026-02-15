use std::fmt::Debug;
use crate::util::*;

pub struct OwnedDomainMask<const GRID_DIMENSION: usize, DomainType: PartialEq + Eq + Copy + Clone + Debug> {
    aabb: AABB<GRID_DIMENSION>,
    buffer: Vec<DomainType>,
}

impl<const GRID_DIMENSION: usize, DomainType: PartialEq + Eq + Copy + Clone + Debug> OwnedDomainMask<GRID_DIMENSION, DomainType> {
    pub fn new(aabb: AABB<GRID_DIMENSION>, d: DomainType) -> Self {
        let buffer = vec![d; aabb.buffer_size()];
        OwnedDomainMask { aabb, buffer }
    }

    /// Get the AABB for this domain
    pub fn aabb(&self) -> &AABB<GRID_DIMENSION> {
        &self.aabb 
    }

    /// Get the buffer, this will be sliced to the right size for the aabb.
    pub fn buffer(&self) -> &[DomainType] {
        &self.buffer
    }

    /// Get mutable access to the buffer,
    /// this will be sliced to the right size for the aabb.
    pub fn buffer_mut(&mut self) -> &mut [DomainType] {
        &mut self.buffer
    }

    /// Access the value at tbe given world coord.
    #[track_caller]
    pub fn view(&self, world_coord: &Coord<GRID_DIMENSION>) -> DomainType {
        debug_assert!(
            self.aabb.contains(world_coord),
            "{:?} does not contain {:?}",
            self.aabb,
            world_coord
        );
        let index = self.aabb.coord_to_linear(world_coord);
        self.buffer[index]
    }

    #[track_caller]
    pub fn set_coord(&mut self, world_coord: &Coord<GRID_DIMENSION>, value: DomainType) {
        debug_assert!(
            self.aabb.contains(world_coord),
            "{:?} does not contain {:?}",
            self.aabb,
            world_coord
        );
        let index = self.aabb.coord_to_linear(world_coord);
        self.buffer[index] = value;
    }
}



