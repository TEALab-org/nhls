use crate::domain::*;
use crate::util::*;

/// Generate normal like distribution over bound with spike in the middle,
/// all values are in [0, 1].
pub fn normal_ic_1d<DomainType: DomainView<1>>(
    domain: &mut DomainType,
    variance: f64,
    chunk_size: usize,
) {
    let n_f = domain.aabb().buffer_size() as f64;
    let sigma_sq: f64 = (n_f / variance) * (n_f / variance);
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
    variance: f64,
    chunk_size: usize,
) {
    let exclusive_bounds = domain.aabb().exclusive_bounds();
    let width_f = exclusive_bounds[0] as f64;
    let height_f = exclusive_bounds[1] as f64;
    let sigma_sq: f64 = (width_f / variance) * (width_f / variance);
    let ic_gen = |coord: Coord<2>| {
        let x = (coord[0] as f64) - (width_f / 2.0);
        let y = (coord[1] as f64) - (height_f / 2.0);
        let r = (x * x + y * y).sqrt();
        let exp = -r * r / (2.0 * sigma_sq);
        exp.exp()
    };
    domain.par_set_values(ic_gen, chunk_size);
}

/// Generate normal like distribution over bound with spike in the middle,
/// all values are in [0, 1].
pub fn normal_ic_3d<DomainType: DomainView<3>>(
    domain: &mut DomainType,
    variance: f64,
    chunk_size: usize,
) {
    let exclusive_bounds = domain.aabb().exclusive_bounds();
    let width_f = exclusive_bounds[0] as f64;
    let height_f = exclusive_bounds[1] as f64;
    let depth_f = exclusive_bounds[2] as f64;
    let sigma_sq: f64 = (width_f / variance) * (width_f / variance);
    let ic_gen = |coord: Coord<3>| {
        let x = (coord[0] as f64) - (width_f / 2.0);
        let y = (coord[1] as f64) - (height_f / 2.0);
        let z = (coord[2] as f64) - (depth_f / 2.0);
        let r = (x * x + y * y + z * z).sqrt();
        let exp = -r * r / (2.0 * sigma_sq);
        exp.exp()
    };
    domain.par_set_values(ic_gen, chunk_size);
}
