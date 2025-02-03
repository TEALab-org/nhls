mod constant;
mod periodic;

pub use constant::*;
pub use periodic::*;

use crate::util::*;

pub trait BCCheck<const GRID_DIMENSION: usize>: Sync {
    fn check(
        &self,
        world_coord: &Coord<GRID_DIMENSION>,
        global_time: usize,
    ) -> Option<f64>;
}
