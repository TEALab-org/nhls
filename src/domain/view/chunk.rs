use crate::util::*;

pub struct DomainChunk<'a, const GRID_DIMENSION: usize> {
    offset: usize,
    aabb: &'a AABB<GRID_DIMENSION>,
    buffer: &'a mut [f64],
}

impl<'a, const GRID_DIMENSION: usize> DomainChunk<'a, GRID_DIMENSION> {
    pub fn new(
        offset: usize,
        aabb: &'a AABB<GRID_DIMENSION>,
        buffer: &'a mut [f64],
    ) -> Self {
        DomainChunk {
            offset,
            aabb,
            buffer,
        }
    }

    pub fn coord_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = (Coord<GRID_DIMENSION>, &mut f64)> {
        self.buffer
            .iter_mut()
            .enumerate()
            .map(|(i, v): (usize, &mut f64)| {
                let linear_index = self.offset + i;
                let coord = self.aabb.linear_to_coord(linear_index);
                (coord, v)
            })
    }
}
