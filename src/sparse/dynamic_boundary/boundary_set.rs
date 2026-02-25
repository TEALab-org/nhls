use crate::util::*;

use std::collections::HashSet;

#[derive(Clone)]
pub struct BoundarySet<const GRID_DIMENSION: usize> {
    pub cells: HashSet<Coord<GRID_DIMENSION>>,
    aabb: AABB<GRID_DIMENSION>,
}

impl<const GRID_DIMENSION: usize> BoundarySet<GRID_DIMENSION> {
    pub fn empty() -> Self {
        BoundarySet {
            cells: HashSet::new(),
            aabb: AABB::empty(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn add(&mut self, coord: Coord<GRID_DIMENSION>) {
        self.aabb.add_coord(&coord);
        self.cells.insert(coord);
    }

    pub fn contains(&self, coord: &Coord<GRID_DIMENSION>) -> bool {
        self.cells.contains(coord)
    }

    pub fn remove(&mut self, coord: &Coord<GRID_DIMENSION>) {
        self.cells.remove(coord);
    }

    pub fn coord_iter(&self) -> impl Iterator<Item = &Coord<GRID_DIMENSION>> {
        self.cells.iter()
    }

    pub fn clear(&mut self) {
        self.cells.clear();
        self.aabb = AABB::empty();
    }

    pub fn combine(&self, other: &Self) -> Self {
        let mut cells = self.cells.clone();
        cells.extend(other.cells.iter());
        let mut aabb = self.aabb;
        aabb.add_aabb(&other.aabb);
        Self { cells, aabb }
    }
}
