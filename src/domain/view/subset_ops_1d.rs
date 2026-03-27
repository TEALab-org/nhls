use crate::domain::view::*;

pub struct SubsetOps1d {}

impl SubsetOps<1> for SubsetOps1d {
    fn copy<DomainType: DomainView<1>>(
        &self,
        src: &DomainType,
        dst: &mut DomainType,
        aabb: &AABB<1>,
        _threads: usize,
    ) {
        profiling::scope!("subsetops1d::copy");
        debug_assert!(src.aabb().contains_aabb(aabb));
        debug_assert!(dst.aabb().contains_aabb(aabb));

        let aabb_ex_b = aabb.exclusive_bounds();
        let row_width = aabb_ex_b[0] as usize;

        let src_origin = src.aabb().coord_to_linear(&aabb.min());
        let dst_origin = dst.aabb().coord_to_linear(&aabb.min());

        let src_end_index = src_origin + row_width;
        let dst_end_index = dst_origin + row_width;

        let src_slice = &src.buffer()[src_origin..src_end_index];
        let dst_slice = &mut dst.buffer_mut()[dst_origin..dst_end_index];
        dst_slice.copy_from_slice(src_slice);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn subdomain_1d() {
        let threads = 2;
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
        ops.copy(
            &bigger_domain,
            &mut smaller_domain,
            &smaller_domain_bounds,
            threads,
        );
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

        ops.copy(
            &smaller_domain,
            &mut bigger_domain,
            &smaller_domain_bounds,
            threads,
        );
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
