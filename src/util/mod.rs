pub mod indexing;

mod aabb;
pub use aabb::*;
pub use fftw::array::AlignedVec;
pub use nalgebra::{matrix, vector};

pub use num_traits::{Num, One, Zero};

pub use fftw::array::*;
pub use fftw::types::c64;

pub trait NumTrait = Num + Copy + Send + Sync;

pub type Coord<const GRID_DIMENSION: usize> =
    nalgebra::SVector<i32, { GRID_DIMENSION }>;

/// Raw type used for AABB,
/// an n by 2 matrix, where
/// column 0 is the min corner
/// and column 1 is the max corner
pub type Bounds<const DIMENSION: usize> =
    nalgebra::SMatrix<i32, { DIMENSION }, 2>;

pub type Values<const NEIGHBORHOOD_SIZE: usize> =
    nalgebra::SMatrix<f64, { NEIGHBORHOOD_SIZE }, 1>;

#[inline]
pub fn flip_sloped<const GRID_DIMENSION: usize>(
    sloped: &Bounds<GRID_DIMENSION>,
) -> Bounds<GRID_DIMENSION> {
    debug_assert!(sloped.min() >= 0);
    debug_assert!(sloped.max() <= 1);
    sloped.add_scalar(-1) * -1
}

#[inline]
pub fn slopes_to_outward_diff<const GRID_DIMENSION: usize>(
    slopes: &Bounds<GRID_DIMENSION>,
) -> Bounds<GRID_DIMENSION> {
    let mut diff_slopes = *slopes;
    let negative_slioes = -1 * diff_slopes.column(0);
    diff_slopes.set_column(0, &negative_slioes);
    diff_slopes
}

pub fn slopes_to_circ_aabb<const GRID_DIMENSION: usize>(
    slopes: &Bounds<GRID_DIMENSION>,
) -> AABB<GRID_DIMENSION> {
    let total_width: Coord<GRID_DIMENSION> =
        slopes.column(0) + slopes.column(1);
    let mut domain_bounds: Bounds<GRID_DIMENSION> = Bounds::zero();
    domain_bounds.set_column(1, &total_width);
    AABB::new(domain_bounds)
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn flip_slopes_test() {
        {
            let slopes = matrix![0, 1];
            debug_assert_eq!(flip_sloped(&slopes), matrix![1, 0]);
        }

        {
            let slopes = matrix![0, 1; 1, 1; 0, 0; 1, 0];
            debug_assert_eq!(
                flip_sloped(&slopes),
                matrix![1, 0; 0, 0; 1, 1; 0, 1]
            );
        }
    }

    #[test]
    fn slopes_to_diff() {
        {}
    }
}
