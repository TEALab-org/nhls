use crate::domain::*;
use rand::prelude::*;
use rayon::prelude::*;

/// This matches the init behaivor of the 2023 implementation
pub fn rand_ic<
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
