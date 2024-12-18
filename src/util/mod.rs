pub use num_traits::{Num, One, Zero};

pub trait NumTrait = Num + Copy + Send + Sync;

mod aabb;
pub mod indexing;
pub use aabb::*;

pub type Coord<const GRID_DIMENSION: usize> = nalgebra::SVector<i32, { GRID_DIMENSION }>;
