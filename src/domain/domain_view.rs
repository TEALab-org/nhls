use crate::util::*;
use rayon::prelude::*;

pub struct Domain<'a, const GRID_DIMENSION: usize> {
    aabb: AABB<GRID_DIMENSION>,
    buffer: &'a mut [f32],
}

impl<'a, const GRID_DIMENSION: usize> Domain<'a, GRID_DIMENSION> {
    pub fn aabb(&self) -> &AABB<GRID_DIMENSION> {
        &self.aabb
    }

    pub fn set_aabb(&mut self, aabb: AABB<GRID_DIMENSION>) {
        debug_assert!(aabb.buffer_size() <= self.buffer.len());
        // TODO: should we re-slice here?
        self.aabb = aabb;
    }

    pub fn buffer(&self) -> &[f32] {
        self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut [f32] {
        self.buffer
    }

    pub fn new(aabb: AABB<GRID_DIMENSION>, buffer: &'a mut [f32]) -> Self {
        debug_assert_eq!(buffer.len(), aabb.buffer_size());
        Domain { aabb, buffer }
    }

    pub fn view(&self, world_coord: &Coord<GRID_DIMENSION>) -> f32 {
        debug_assert!(self.aabb.contains(world_coord));
        let index = self.aabb.coord_to_linear(world_coord);
        self.buffer[index]
    }

    pub fn modify(&mut self, world_coord: &Coord<GRID_DIMENSION>, value: f32) {
        debug_assert!(self.aabb.contains(world_coord));
        let index = self.aabb.coord_to_linear(world_coord);
        self.buffer[index] = value;
    }

    pub fn par_modify_access(
        &mut self,
        chunk_size: usize,
    ) -> impl ParallelIterator<Item = DomainChunk<'_, GRID_DIMENSION>> {
        par_modify_access_impl(self.buffer, &self.aabb, chunk_size)
    }

    pub fn par_set_values<
        F: FnOnce(Coord<GRID_DIMENSION>) -> f32 + Send + Sync + Copy,
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
                        &mut f32,
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
    buffer: &'a mut [f32],
    aabb: &'a AABB<GRID_DIMENSION>,
    chunk_size: usize,
) -> impl ParallelIterator<Item = DomainChunk<'a, GRID_DIMENSION>> + 'a {
    buffer[0..aabb.buffer_size()]
        .par_chunks_mut(chunk_size)
        .enumerate()
        .map(move |(i, buffer_chunk): (usize, &mut [f32])| {
            let offset = i * chunk_size;
            DomainChunk::new(offset, aabb, buffer_chunk)
        })
}

pub struct DomainChunk<'a, const GRID_DIMENSION: usize> {
    offset: usize,
    aabb: &'a AABB<GRID_DIMENSION>,
    buffer: &'a mut [f32],
}

impl<'a, const GRID_DIMENSION: usize> DomainChunk<'a, GRID_DIMENSION> {
    pub fn new(
        offset: usize,
        aabb: &'a AABB<GRID_DIMENSION>,
        buffer: &'a mut [f32],
    ) -> Self {
        DomainChunk {
            offset,
            aabb,
            buffer,
        }
    }

    pub fn coord_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = (Coord<GRID_DIMENSION>, &mut f32)> {
        self.buffer
            .iter_mut()
            .enumerate()
            .map(|(i, v): (usize, &mut f32)| {
                let linear_index = self.offset + i;
                let coord = self.aabb.linear_to_coord(linear_index);
                (coord, v)
            })
    }
}
