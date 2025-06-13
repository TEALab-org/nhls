use crate::domain::view::*;

pub struct SubsetOps2d {}

impl SubsetOps<2> for SubsetOps2d {
    fn copy_to_subdomain<DomainType: DomainView<2>>(
        &self,
        bigger_domain: &DomainType,
        smaller_domain: &mut DomainType,
        _threads: usize,
    ) {
        profiling::scope!("SubsetOps2d::copy_to_subdomain");
        debug_assert!(bigger_domain
            .aabb()
            .contains_aabb(smaller_domain.aabb()));

        // For axis 0, slice copy y axis
        // I think we can also update bigger_min_index with some constant offet?
        // bigger width of course!

        // herer though smaller width is the y-axis size for
        let smaller_exclusive_bounds = smaller_domain.aabb().exclusive_bounds();

        let smaller_min = smaller_domain.aabb().min();
        let mut bigger_min_index =
            bigger_domain.aabb().coord_to_linear(&smaller_min);
        let mut smaller_min_index = 0;
        let smaller_width = smaller_exclusive_bounds[1] as usize;

        let bigger_bounds = bigger_domain.aabb().bounds;
        let bigger_width =
            (bigger_bounds[(1, 1)] - bigger_bounds[(1, 0)]) as usize + 1;

        for _ in 0..smaller_exclusive_bounds[0] {
            smaller_domain.buffer_mut()
                [smaller_min_index..smaller_min_index + smaller_width]
                .copy_from_slice(
                    &bigger_domain.buffer()
                        [bigger_min_index..bigger_min_index + smaller_width],
                );
            bigger_min_index += bigger_width;
            smaller_min_index += smaller_width;
        }
    }

    fn copy_from_subdomain<DomainType: DomainView<2>>(
        &self,
        smaller_domain: &DomainType,
        bigger_domain: &mut DomainType,
        _threads: usize,
    ) {
        profiling::scope!("SubsetOps2d::copy_from_subdomain");
        debug_assert!(bigger_domain
            .aabb()
            .contains_aabb(smaller_domain.aabb()));

        // For axis 0, slice copy y axis
        // I think we can also update bigger_min_index with some constant offet?
        // bigger width of course!

        // herer though smaller width is the y-axis size for
        let smaller_exclusive_bounds = smaller_domain.aabb().exclusive_bounds();

        let smaller_min = smaller_domain.aabb().min();
        let mut bigger_min_index =
            bigger_domain.aabb().coord_to_linear(&smaller_min);
        let mut smaller_min_index = 0;
        let smaller_width = smaller_exclusive_bounds[1] as usize;

        let bigger_bounds = bigger_domain.aabb().bounds;
        let bigger_width =
            (bigger_bounds[(1, 1)] - bigger_bounds[(1, 0)]) as usize + 1;

        for _ in 0..smaller_exclusive_bounds[0] {
            bigger_domain.buffer_mut()
                [bigger_min_index..bigger_min_index + smaller_width]
                .copy_from_slice(
                    &smaller_domain.buffer()
                        [smaller_min_index..smaller_min_index + smaller_width],
                );
            bigger_min_index += bigger_width;
            smaller_min_index += smaller_width;
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn subdomain_2d() {
        let chunk_size = 10;
        let bigger_domain_bounds = AABB::new(matrix![0, 9; 0, 9]);
        let mut bigger_domain = OwnedDomain::new(bigger_domain_bounds);

        let smaller_domain_bounds = AABB::new(matrix![3, 7; 3, 7]);
        let mut smaller_domain = OwnedDomain::new(smaller_domain_bounds);

        bigger_domain.par_set_values(|_| 1.0, chunk_size);
        smaller_domain.par_set_values(|_| 2.0, chunk_size);

        // Bigger domain should be same,
        // smaller domain should be 1s
        let ops = SubsetOps2d {};
        ops.copy_to_subdomain(&bigger_domain, &mut smaller_domain);
        for x in 0..=9 {
            for y in 0..=9 {
                assert_eq!(bigger_domain.view(&vector![x, y]), 1.0);
            }
        }
        for x in 3..=7 {
            for y in 3..=7 {
                assert_eq!(smaller_domain.view(&vector![x, y]), 1.0);
            }
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
        ops.copy_from_subdomain(&smaller_domain, &mut bigger_domain);
        for x in 0..=9 {
            for y in 0..=9 {
                if (3..=7).contains(&x) && (3..=7).contains(&y) {
                    assert_eq!(bigger_domain.view(&vector![x, y]), 2.0);
                    assert_eq!(smaller_domain.view(&vector![x, y]), 2.0);
                } else {
                    assert_eq!(bigger_domain.view(&vector![x, y]), 1.0);
                }
            }
        }
    }
}
