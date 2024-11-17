use crate::util::*;
use rayon::prelude::*;

pub struct Domain<'a, const GRID_DIMENSION: usize> {
    view_box: Box<GRID_DIMENSION>,
    buffer: &'a mut [f32],
}

impl<'a, const GRID_DIMENSION: usize> Domain<'a, GRID_DIMENSION> {
    pub fn view_box(&self) -> &Box<GRID_DIMENSION> {
        &self.view_box
    }

    pub fn new(view_box: Box<GRID_DIMENSION>, buffer: &'a mut [f32]) -> Self {
        debug_assert_eq!(buffer.len(), box_buffer_size(&view_box));
        Domain { view_box, buffer }
    }

    pub fn view(&self, world_coord: &Coord<GRID_DIMENSION>) -> f32 {
        debug_assert!(coord_in_box(world_coord, &self.view_box));
        let index = coord_to_linear_in_box(world_coord, &self.view_box);
        println!(
            "view, c: {:?}, i: {}, r: {}",
            world_coord, index, self.buffer[index]
        );
        self.buffer[index]
    }

    pub fn modify(&mut self, world_coord: &Coord<GRID_DIMENSION>, value: f32) {
        debug_assert!(coord_in_box(world_coord, &self.view_box));
        let index = coord_to_linear_in_box(world_coord, &self.view_box);
        self.buffer[index] = value;
    }

    pub fn par_modify_access(
        &mut self,
        chunk_size: usize,
    ) -> impl ParallelIterator<Item = DomainChunk<'_, GRID_DIMENSION>> {
        par_modify_access_impl(self.buffer, &self.view_box, chunk_size)
    }
}

/// Why not just put this into Domain::par_modify_access?
/// Rust compiler can't figure out how to borrow view_box and buffer
/// at the same time in this way.
/// By putting their borrows into one function call first we work around it.
fn par_modify_access_impl<'a, const GRID_DIMENSION: usize>(
    buffer: &'a mut [f32],
    view_box: &'a Box<GRID_DIMENSION>,
    chunk_size: usize,
) -> impl ParallelIterator<Item = DomainChunk<'a, GRID_DIMENSION>> + 'a {
    buffer.par_chunks_mut(chunk_size).enumerate().map(
        move |(i, buffer_chunk): (usize, &mut [f32])| {
            let offset = i * chunk_size;
            DomainChunk::new(offset, view_box, buffer_chunk)
        },
    )
}

pub struct DomainChunk<'a, const GRID_DIMENSION: usize> {
    offset: usize,
    view_box: &'a Box<GRID_DIMENSION>,
    buffer: &'a mut [f32],
}

impl<'a, const GRID_DIMENSION: usize> DomainChunk<'a, GRID_DIMENSION> {
    pub fn new(offset: usize, view_box: &'a Box<GRID_DIMENSION>, buffer: &'a mut [f32]) -> Self {
        DomainChunk {
            offset,
            view_box,
            buffer,
        }
    }

    pub fn coord_iter_mut(&mut self) -> impl Iterator<Item = (Coord<GRID_DIMENSION>, &mut f32)> {
        self.buffer
            .iter_mut()
            .enumerate()
            .map(|(i, v): (usize, &mut f32)| {
                let linear_index = self.offset + i;
                let coord = linear_to_coord_in_box(linear_index, self.view_box);
                (coord, v)
            })
    }
}

/*
// parallel stencil application
// One owning domain gets chunked out

pub struct DomainView<'a, const GRID_DIMENSION: usize> {
    // view box
    view_box: &'a Box<GRID_DIMENSION>,

    // input buffer
    buffer: &'a [f32],
}

// ah but we want to mutate and not
pub struct DomainViewMut<'a, const GRID_DIMENSION: usize> {
    view_box: &'a Box<GRID_DIMENSION>,
    buffer: &'a mut [f32],
}
*/
