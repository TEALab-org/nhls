use crate::domain::view::*;

pub trait SubsetOps<const GRID_DIMENSION: usize>: Sync {
    fn copy<DomainType: DomainView<GRID_DIMENSION>>(
        &self,
        src: &DomainType,
        dst: &mut DomainType,
        aabb: &AABB<GRID_DIMENSION>,
        threads: usize,
    );
}
