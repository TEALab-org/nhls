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

pub struct SubsetOps1d {}

impl SubsetOps<1> for SubsetOps1d {
    fn copy_to_subdomain<DomainType: DomainView<1>>(
        &self,
        src_domain: &DomainType,
        dst_domain: &mut DomainType,
    ) {
        debug_assert!(src_domain.aabb().contains_aabb(dst_domain.aabb()));
        // Get src domain slice, whole thing
        let dst_size = dst_domain.aabb().buffer_size();
        let dst_min = dst_domain.aabb().min();
        let src_min_index = src_domain.aabb().coord_to_linear(&dst_min);
        dst_domain
            .buffer_mut()
            .copy_from_slice(&src_domain.buffer()[src_min_index..src_min_index + dst_size]);
    }

    fn copy_from_subdomain<DomainType: DomainView<1>>(
        &self,
        src_domain: &DomainType,
        dst_domain: &mut DomainType,
    ) {
        debug_assert!(dst_domain.aabb().contains_aabb(src_domain.aabb()));
        let src_size = src_domain.aabb().buffer_size();
        let src_min = src_domain.aabb().min();
        let dst_min_index = dst_domain.aabb().coord_to_linear(&src_min);
        dst_domain
            .buffer_mut()[dst_min_index..dst_min_index + src_size]
            .copy_from_slice(src_domain.buffer());
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn subdomain_1d() {
        let chunk_size = 10;
        let bigger_domain_bounds = AABB::new(matrix![0, 9]);
        let mut bigger_domain = OwnedDomain::new(bigger_domain_bounds);

        let smaller_domain_bounds = AABB::new(matrix![3, 7]);
        let mut smaller_domain = OwnedDomain::new(smaller_domain_bounds);

        bigger_domain.par_set_values(|_| 1.0, chunk_size);
        smaller_domain.par_set_values(|_| 2.0, chunk_size);

        // Bigger domain should be same,
        // smaller domain should be 1s
        let ops = SubsetOps1d {};
        ops.copy_to_subdomain(&bigger_domain, &mut smaller_domain);
        for i in 0..=9 {
            assert_eq!(bigger_domain.view(&vector![i]), 1.0);
        }
        for i in 3..=7 {
            assert_eq!(smaller_domain.view(&vector![i]), 1.0);
        }

        // Double check that with coord iters
        for c in bigger_domain.aabb().coord_iter() {
           assert_eq!(bigger_domain.view(&c), 1.0); 
        }
        for c in smaller_domain.aabb().coord_iter() {
           assert_eq!(bigger_domain.view(&c), 1.0); 
        }

        // Smaller domain should be all twos
        // bigger domain should have some twos too
        smaller_domain.par_set_values(|_| 2.0, chunk_size);
        bigger_domain.par_set_values(|_| 1.0, chunk_size);

        let ops = SubsetOps1d {};
        ops.copy_from_subdomain(&smaller_domain, &mut bigger_domain, );
        for i in 0..=9 {
            if (3..=7).contains(&i) {
                assert_eq!(bigger_domain.view(&vector![i]), 2.0);
                assert_eq!(smaller_domain.view(&vector![i]), 2.0);
            } else {
                assert_eq!(bigger_domain.view(&vector![i]), 1.0);
            }
        }
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
