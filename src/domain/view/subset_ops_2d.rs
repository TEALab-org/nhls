use crate::domain::view::*;
use rayon::prelude::*;

pub struct SubsetOps2d {
    pub chunk_size: usize,
}

impl SubsetOps<2> for SubsetOps2d {
    fn copy<DomainType: DomainView<2>>(
        &self,
        src: &DomainType,
        dst: &mut DomainType,
        aabb: &AABB<2>,
        threads: usize,
    ) {
        profiling::scope!("subsetops2d::copy");
        debug_assert!(src.aabb().contains_aabb(aabb));
        debug_assert!(dst.aabb().contains_aabb(aabb));

        let aabb_ex_b = aabb.exclusive_bounds();
        let src_ex_b = src.aabb().exclusive_bounds();
        let dst_ex_b = dst.aabb().exclusive_bounds();

        let row_width = aabb_ex_b[1] as usize;
        let src_width = src_ex_b[1] as usize;
        let dst_width = dst_ex_b[1] as usize;

        let src_origin = src.aabb().coord_to_linear(&aabb.min());
        let dst_origin = dst.aabb().coord_to_linear(&aabb.min());

        let height = aabb_ex_b[0] as usize;
        let chunk_size = self.chunk_size.max(height / threads);
        let chunks = height.div_ceil(chunk_size);

        (0..chunks).into_par_iter().for_each(move |c| {
            profiling::scope!("subsetops2d::copy Thread callback");

            let mut src_index = src_origin + c * chunk_size * src_width;
            let mut dst_index = dst_origin + c * chunk_size * dst_width;

            let start_row = c * chunk_size;
            let end_row = height.min((c + 1) * chunk_size);
            //println!("THREAD: start_i: {start_i}, end_i: {end_i}");
            let mut dst_t = dst.unsafe_mut_access();
            for _ in start_row..end_row {
                let src_end_index = src_index + row_width;
                let dst_end_index = dst_index + row_width;
                //println!("THREAD: copy {smaller_min_index_t} -> {smaller_end_index}");
                let src_slice = &src.buffer()[src_index..src_end_index];
                let dst_slice =
                    &mut dst_t.buffer_mut()[dst_index..dst_end_index];
                dst_slice.copy_from_slice(src_slice);

                src_index += src_width;
                dst_index += dst_width;
            }
        });
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn subdomain_2d() {
        let chunk_size = 10;
        let threads = 2;
        let bigger_domain_bounds = AABB::new(matrix![0, 9; 0, 9]);
        let mut bigger_domain = OwnedDomain::new(bigger_domain_bounds);

        let smaller_domain_bounds = AABB::new(matrix![3, 7; 3, 7]);
        let mut smaller_domain = OwnedDomain::new(smaller_domain_bounds);

        bigger_domain.par_set_values(|_| 1.0, chunk_size);
        smaller_domain.par_set_values(|_| 2.0, chunk_size);

        // Bigger domain should be same,
        // smaller domain should be 1s
        let ops = SubsetOps2d { chunk_size };
        let aabb = *smaller_domain.aabb();
        ops.copy(&bigger_domain, &mut smaller_domain, &aabb, threads);
        //println!("Smaller after:");
        //print_debug(&smaller_domain);

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
        ops.copy(&smaller_domain, &mut bigger_domain, &aabb, threads);
        //ops.copy_from_subdomain(&smaller_domain, &mut bigger_domain, threads);
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
