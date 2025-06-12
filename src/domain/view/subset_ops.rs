use crate::domain::view::*;

pub trait SubsetOps<const GRID_DIMENSION: usize>: Sync {
    /// src_domain is a sub-set of target domain,
    /// copy values from src into target
    fn copy_to_subdomain<DomainType: DomainView<GRID_DIMENSION>>(
        &self,
        src_domain: &DomainType,
        target_domain: &mut DomainType,
    );

    /// target domain is a sub-set of source domain,
    /// copy values from src into target
    /// effectivley init all of target domain
    fn copy_from_subdomain<DomainType: DomainView<GRID_DIMENSION>>(
        &self,
        src_domain: &DomainType,
        target_domain: &mut DomainType,
    );
}
