//! Domain Initilization
//!
//! Utilities for common domain initilization.
//! Use `DomainView::par_set_values` for custom needs.

use crate::domain::*;
use crate::util::*;
use rand::prelude::*;
use rayon::prelude::*;

/// This matches the init behaivor of the 2023 implementation
pub fn rand<
    const GRID_DIMENSION: usize,
    DomainType: DomainView<GRID_DIMENSION>,
>(
    domain: &mut DomainType,
    max_val: i32,
    chunk_size: usize,
) {
    domain.par_modify_access(chunk_size).for_each(
        |mut d: DomainChunk<'_, GRID_DIMENSION>| {
            let mut rng = rand::thread_rng();
            d.coord_iter_mut().for_each(|(_, value_mut)| {
                *value_mut = (rng.gen::<i32>() % max_val) as f64;
            })
        },
    );
}

/// Generate normal like distribution over bound with spike in the middle,
/// all values are in [0, 1].
pub fn normal_ic_1d<DomainType: DomainView<1>>(
    domain: &mut DomainType,
    chunk_size: usize,
) {
    let n_f = domain.aabb().buffer_size() as f64;
    let sigma_sq: f64 = (n_f / 25.0) * (n_f / 25.0);
    let ic_gen = |world_coord: Coord<1>| {
        let x = (world_coord[0] as f64) - (n_f / 2.0);
        let exp = -x * x / (2.0 * sigma_sq);
        exp.exp()
    };
    domain.par_set_values(ic_gen, chunk_size);
}

/// Generate normal like distribution over bound with spike in the middle,
/// all values are in [0, 1].
pub fn normal_ic_2d<DomainType: DomainView<2>>(
    domain: &mut DomainType,
    chunk_size: usize,
) {
    let exclusive_bounds = domain.aabb().exclusive_bounds();
    let width_f = exclusive_bounds[0] as f64;
    let height_f = exclusive_bounds[1] as f64;
    let sigma_sq: f64 = (width_f / 25.0) * (width_f / 25.0);
    let ic_gen = |coord: Coord<2>| {
        let x = (coord[0] as f64) - (width_f / 2.0);
        let y = (coord[1] as f64) - (height_f / 2.0);
        let r = (x * x + y * y).sqrt();
        let exp = -r * r / (2.0 * sigma_sq);
        exp.exp()
    };
    domain.par_set_values(ic_gen, chunk_size);
}
