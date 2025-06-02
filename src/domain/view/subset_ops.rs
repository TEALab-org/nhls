use crate::domain::view::*;

pub trait SubsetOps<const GRID_DIMENSION: usize>: Sync {
    fn set_subdomain(
        domain: &SliceDomain<GRID_DIMENSION>,
        subdomain: &mut SliceDomain<GRID_DIMENSION>,
    );

    fn from_subdomain(
        domain: &mut SliceDomain<GRID_DIMENSION>,
        subdomain: &SliceDomain<GRID_DIMENSION>,
    );
}
