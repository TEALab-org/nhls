mod chunk;
mod owned;
mod slice;

pub use chunk::*;
pub use owned::*;
pub use slice::*;

use crate::util::*;
use rayon::prelude::*;

pub trait DomainView<const GRID_DIMENSION: usize> {
    fn aabb(&self) -> &AABB<GRID_DIMENSION>;

    fn set_aabb(&mut self, aabb: AABB<GRID_DIMENSION>);

    fn buffer(&self) -> &[f64];

    fn buffer_mut(&mut self) -> (&AABB<GRID_DIMENSION>, &mut [f64]);

    fn view(&self, world_coord: &Coord<GRID_DIMENSION>) -> f64;

    fn par_modify_access<'a>(
        &'a mut self,
        chunk_size: usize,
    ) -> impl ParallelIterator<Item = DomainChunk<'a, GRID_DIMENSION>> {
        let (aabb, buffer) = self.buffer_mut();
        par_modify_access_impl(buffer, aabb, chunk_size)
    }

    fn par_set_values<
        F: FnOnce(Coord<GRID_DIMENSION>) -> f64 + Send + Sync + Copy,
    >(
        &mut self,
        f: F,
        chunk_size: usize,
    ) {
        self.par_modify_access(chunk_size).for_each(
            |mut d: DomainChunk<'_, GRID_DIMENSION>| {
                d.coord_iter_mut().for_each(
                    |(world_coord, value_mut): (
                        Coord<GRID_DIMENSION>,
                        &mut f64,
                    )| {
                        *value_mut = f(world_coord);
                    },
                )
            },
        );
    }
}

/// Why not just put this into Domain::par_modify_access?
/// Rust compiler can't figure out how to borrow aabb and buffer
/// at the same time in this way.
/// By putting their borrows into one function call first we work around it.
fn par_modify_access_impl<'a, const GRID_DIMENSION: usize>(
    buffer: &'a mut [f64],
    aabb: &'a AABB<GRID_DIMENSION>,
    chunk_size: usize,
) -> impl ParallelIterator<Item = DomainChunk<'a, GRID_DIMENSION>> + 'a {
    buffer[0..aabb.buffer_size()]
        .par_chunks_mut(chunk_size)
        .enumerate()
        .map(move |(i, buffer_chunk): (usize, &mut [f64])| {
            let offset = i * chunk_size;
            DomainChunk::new(offset, aabb, buffer_chunk)
        })
}
