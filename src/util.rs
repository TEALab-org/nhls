pub use num_traits::{Num, One, Zero};

pub trait NumTrait = Num + Copy + Send + Sync;

pub type Bound<const GRID_DIMENSION: usize> = nalgebra::SVector<i32, { GRID_DIMENSION }>;
pub type Slopes<const GRID_DIMENSION: usize> = nalgebra::SMatrix<i32, { GRID_DIMENSION }, 2>;
