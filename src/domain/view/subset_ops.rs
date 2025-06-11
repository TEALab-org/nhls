use crate::domain::view::*;

pub trait SubsetOps<const GRID_DIMENSION: usize>: Sync {
    /// src_domain is a sub-set of target domain,
    /// copy values from src into target
    fn set_subdomain(
        src_domain: &SliceDomain<GRID_DIMENSION>,
        target_domain: &mut SliceDomain<GRID_DIMENSION>,
    );

    /// target domain is a sub-set of source domain,
    /// copy values from src into target
    /// effectivley init all of target domain
    fn from_subdomain(
        src_domain: &SliceDomain<GRID_DIMENSION>,
        target_domain: &mut SliceDomain<GRID_DIMENSION>,
    );
}

pub struct SubsetOps1d {}

impl SubsetOps<1> for SubsetOps1d {
    fn set_subdomain(
            src_domain: &SliceDomain<1>,
            target_domain: &mut SliceDomain<1>,
        ) {
        
    }

    fn from_subdomain(
            src_domain: & SliceDomain<1>,
            target_domain: &mut SliceDomain<1>,
        ) {
       // block copy 
    }
}

/*
pub struct SubsetOps2d {}

impl SubsetOps<2> for SubsetOps2d {
    fn set_subdomain(
            domain: &SliceDomain<2>,
            subdomain: &mut SliceDomain<2>,
        ) {
        // Along y axis, block copy x axis
    }

    fn from_subdomain(
            domain: &mut SliceDomain<2>,
            subdomain: &SliceDomain<2>,
        ) {
       // Along y axis block copy x axis
    }
}
*/
