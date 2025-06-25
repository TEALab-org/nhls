use crate::domain::view::*;

pub struct SubsetOps3d {
    pub chunk_size: usize,
    pub task_min: usize,
}

impl SubsetOps<3> for SubsetOps3d {
    fn copy<DomainType: DomainView<3>>(
        &self,
        src: &DomainType,
        dst: &mut DomainType,
        aabb: &AABB<3>,
        threads: usize,
    ) {
        profiling::scope!("subsetops3d::copy");
        debug_assert!(src.aabb().contains_aabb(aabb));
        debug_assert!(dst.aabb().contains_aabb(aabb));

        let aabb_ex_b = aabb.exclusive_bounds();
        let src_ex_b = src.aabb().exclusive_bounds();
        let dst_ex_b = dst.aabb().exclusive_bounds();

        let row_width = aabb_ex_b[2] as usize;
        let src_width = src_ex_b[2] as usize;
        let dst_width = dst_ex_b[2] as usize;

        let src_layer = (src_ex_b[2] * src_ex_b[1]) as usize;
        let dst_layer = (dst_ex_b[2] * dst_ex_b[1]) as usize;

        let src_origin = src.aabb().coord_to_linear(&aabb.min());
        let dst_origin = dst.aabb().coord_to_linear(&aabb.min());

        let height = aabb_ex_b[0] as usize;
        let chunk_size = if threads * 2 < height {
            height / threads
        } else if self.task_min < height {
            height / self.task_min
        } else {
            height
        };

        let chunks = height.div_ceil(chunk_size);

        (0..chunks).into_par_iter().for_each(move |c| {
            profiling::scope!("subsetops3d::copy Thread callback");
            let mut src_index = src_origin + c * chunk_size * src_layer;
            let mut dst_index = dst_origin + c * chunk_size * dst_layer;

            let start_row = c * chunk_size;
            let end_row = height.min((c + 1) * chunk_size);

            let mut dst_t = dst.unsafe_mut_access();
            for _z in start_row..end_row {
                let mut src_row_index = src_index;
                let mut dst_row_index = dst_index;
                for _y in 0..aabb_ex_b[1] {
                    let src_end_index = src_row_index + row_width;
                    let dst_end_index = dst_row_index + row_width;

                    let src_slice = &src.buffer()[src_row_index..src_end_index];
                    let dst_slice =
                        &mut dst_t.buffer_mut()[dst_row_index..dst_end_index];
                    dst_slice.copy_from_slice(src_slice);

                    src_row_index += src_width;
                    dst_row_index += dst_width;
                }
                src_index += src_layer;
                dst_index += dst_layer
            }
        });
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn subdomain_3d() {
        let threads = 1;
        let chunk_size = 10;
        let bigger_domain_bounds = AABB::new(matrix![0, 9; 0, 9; 0, 9]);
        let mut bigger_domain = OwnedDomain::new(bigger_domain_bounds);

        let smaller_domain_bounds = AABB::new(matrix![3, 7; 3, 7; 3, 7]);
        let mut smaller_domain = OwnedDomain::new(smaller_domain_bounds);

        bigger_domain.par_set_values(|_| 1.0, chunk_size);
        smaller_domain.par_set_values(|_| 2.0, chunk_size);

        // Bigger domain should be same,
        // smaller domain should be 1s
        let ops = SubsetOps3d { chunk_size };
        ops.copy(
            &bigger_domain,
            &mut smaller_domain,
            &smaller_domain_bounds,
            threads,
        );
        for x in 0..=9 {
            for y in 0..=9 {
                for z in 0..=9 {
                    assert_eq!(bigger_domain.view(&vector![x, y, z]), 1.0);
                }
            }
        }
        for x in 3..=7 {
            for y in 3..=7 {
                for z in 3..=7 {
                    assert_eq!(smaller_domain.view(&vector![x, y, z]), 1.0);
                }
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
        ops.copy(
            &smaller_domain,
            &mut bigger_domain,
            &smaller_domain_bounds,
            threads,
        );
        for x in 0..=9 {
            for y in 0..=9 {
                for z in 0..=9 {
                    if (3..=7).contains(&x)
                        && (3..=7).contains(&y)
                        && (3..=7).contains(&z)
                    {
                        assert_eq!(bigger_domain.view(&vector![x, y, z]), 2.0);
                        assert_eq!(smaller_domain.view(&vector![x, y, z]), 2.0);
                    } else {
                        assert_eq!(
                            bigger_domain.view(&vector![x, y, z]),
                            1.0,
                            "{:?}, {}",
                            (x, y, z),
                            bigger_domain
                                .aabb()
                                .coord_to_linear(&vector![x, y, z])
                        );
                    }
                }
            }
        }
    }
}
