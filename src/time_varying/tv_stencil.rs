use crate::stencil::*;
use crate::util::*;

pub trait TVStencil<const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>
{
    fn weights(&self, global_time: usize) -> &Values<NEIGHBORHOOD_SIZE>;

    fn offsets(&self) -> &[Coord<GRID_DIMENSION>; NEIGHBORHOOD_SIZE];

    fn slopes(&self) -> Bounds<GRID_DIMENSION> {
        let mut result = Bounds::zero();
        for neighbor in self.offsets() {
            for d in 0..GRID_DIMENSION {
                let neighbor_d = neighbor[d];
                if neighbor_d > 0 {
                    result[(d, 1)] = result[(d, 1)].max(neighbor_d);
                } else {
                    result[(d, 0)] = result[(d, 0)].max(-neighbor_d);
                }
            }
        }
        result
    }
}

impl<const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>
    TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>
    for Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>
{
    fn weights(&self, _global_time: usize) -> &Values<NEIGHBORHOOD_SIZE> {
        self.weights()
    }

    fn offsets(&self) -> &[Coord<GRID_DIMENSION>; NEIGHBORHOOD_SIZE] {
        self.offsets()
    }
}
