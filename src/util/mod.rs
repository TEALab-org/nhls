pub use num_traits::{Num, One, Zero};

pub trait NumTrait = Num + Copy + Send + Sync;

mod aabb;
pub mod polymult;
pub mod micheals_tstencil;
pub mod indexing;
pub use aabb::*;
pub use polymult::*;
pub use micheals_tstencil::*;
pub use fftw::array::AlignedVec;
pub use nalgebra::{matrix, vector};

pub type Coord<const GRID_DIMENSION: usize> =
    nalgebra::SVector<i32, { GRID_DIMENSION }>;
