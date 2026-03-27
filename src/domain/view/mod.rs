mod chunk;
mod debug_io;
mod owned;
mod slice;
mod subset_ops;
mod subset_ops_1d;
mod subset_ops_2d;
mod subset_ops_3d;

pub use chunk::*;
pub use debug_io::*;
pub use owned::*;
pub use slice::*;
pub use subset_ops::*;
pub use subset_ops_1d::*;
pub use subset_ops_2d::*;
pub use subset_ops_3d::*;

use crate::util::*;
use rayon::prelude::*;

pub trait DomainView<const GRID_DIMENSION: usize>: Sync + Send {
    /// Get the AABB for this domain
    fn aabb(&self) -> &AABB<GRID_DIMENSION>;

    /// Set the AABB for this domain,
    /// current buffer must be large enough!
    fn set_aabb(&mut self, aabb: AABB<GRID_DIMENSION>);

    /// Get the buffer, this will be sliced to the right size for the aabb.
    fn buffer(&self) -> &[f64];

    /// Get mutable access to the buffer,
    /// this will be sliced to the right size for the aabb.
    fn buffer_mut(&mut self) -> &mut [f64];

    /// Get both a const reference to the aabb
    /// and a mutable reference to the buffer.
    fn aabb_buffer_mut(&mut self) -> (&AABB<GRID_DIMENSION>, &mut [f64]);

    /// Access the value at tbe given world coord.
    fn view(&self, world_coord: &Coord<GRID_DIMENSION>) -> f64;

    /// Set the value at a given coordinate.
    /// When setting all values in a domain, use par_set_values instead.
    fn set_coord(&mut self, world_coord: &Coord<GRID_DIMENSION>, value: f64);

    fn par_modify_access(
        &mut self,
        chunk_size: usize,
    ) -> impl ParallelIterator<Item = DomainChunk<'_, GRID_DIMENSION>> {
        let (aabb, buffer) = self.aabb_buffer_mut();
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
                profiling::scope!("domain::par_set_values Thread Callback");
                d.coord_iter_mut().for_each(|(world_coord, value_mut)| {
                    *value_mut = f(world_coord);
                })
            },
        );
    }

    fn par_set_from<DomainType: DomainView<GRID_DIMENSION>>(
        &mut self,
        other: &DomainType,
        aabb: &AABB<GRID_DIMENSION>,
        //threads: usize,
    ) {
        profiling::scope!("domain::par_set_from (SINGLE THREADED)");

        for coord in aabb.coord_iter() {
            self.set_coord(&coord, other.view(&coord));
        }
    }

    /// Copy other domain into self
    fn par_set_subdomain<DomainType: DomainView<GRID_DIMENSION>>(
        &mut self,
        other: &DomainType,
        chunk_size: usize,
    ) {
        profiling::scope!("domain::par_set_subdomain");
        let const_self_ref: &Self = self;
        other.buffer()[0..other.aabb().buffer_size()]
            .par_chunks(chunk_size)
            .enumerate()
            .for_each(move |(i, buffer_chunk): (usize, &[f64])| {
                profiling::scope!("domain::par_set_subdomain Thread Callback");
                let self_ptr = const_self_ref as *const Self;
                let mut_self_ref: &mut Self =
                    unsafe { &mut *(self_ptr as *mut Self) as &mut Self };
                let offset = i * chunk_size;
                for i in 0..buffer_chunk.len() {
                    let other_linear_index = i + offset;
                    let world_coord =
                        other.aabb().linear_to_coord(other_linear_index);
                    let self_linear_index =
                        mut_self_ref.aabb().coord_to_linear(&world_coord);
                    mut_self_ref.buffer_mut()[self_linear_index] =
                        other.buffer()[other_linear_index];
                }
            });
    }

    /// Copy self coords from other into self
    fn par_from_superset<DomainType: DomainView<GRID_DIMENSION>>(
        &mut self,
        other: &DomainType,
        chunk_size: usize,
    ) {
        profiling::scope!("domain::par_from_subdomain");
        self.par_set_values(|world_coord| other.view(&world_coord), chunk_size);
    }

    /// WARNING, obviously unsafe.
    ///
    /// In parallel situations, if you can gaurentee that threads are accessing
    /// mutually exclusive coords, then use this as an escape hatch.
    /// DO NOT modify aabb in parallel.
    fn unsafe_mut_access(&self) -> SliceDomain<'_, GRID_DIMENSION> {
        let buffer = self.buffer();
        let len = buffer.len();
        let buffer_ptr = buffer.as_ptr();
        let buffer_ptr_mut = buffer_ptr as *mut f64;
        let unsafe_buffer =
            unsafe { std::slice::from_raw_parts_mut(buffer_ptr_mut, len) };
        SliceDomain::new(*self.aabb(), unsafe_buffer)
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

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn par_set_subdomain_test() {
        {
            let chunk_size = 1;
            let bounds = AABB::new(matrix![0, 9; 0, 9;]);
            let mut domain = OwnedDomain::new(bounds);

            let i_bounds = AABB::new(matrix![3, 7; 3, 7]);
            let mut i_domain = OwnedDomain::new(i_bounds);
            i_domain.par_set_values(|_| 1.0, chunk_size);

            domain.par_set_subdomain(&i_domain, 2);

            for c in domain.aabb().coord_iter() {
                if i_bounds.contains(&c) {
                    assert_eq!(domain.view(&c), 1.0);
                } else {
                    assert_eq!(domain.view(&c), 0.0);
                }
            }
        }
    }

    #[test]
    fn par_from_superset_test() {
        {
            let chunk_size = 1;
            let bounds = AABB::new(matrix![0, 9; 0, 9;]);
            let domain = OwnedDomain::new(bounds);

            let i_bounds = AABB::new(matrix![3, 7; 3, 7]);
            let mut i_domain = OwnedDomain::new(i_bounds);
            i_domain.par_set_values(|_| 1.0, chunk_size);

            i_domain.par_from_superset(&domain, 3);
            for c in i_domain.aabb().coord_iter() {
                assert_eq!(domain.view(&c), 0.0);
            }
        }
    }
}
