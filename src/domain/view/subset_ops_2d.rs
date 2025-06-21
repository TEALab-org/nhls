use crate::domain::view::*;
use rayon::prelude::*;

pub struct SubsetOps2d {
    pub chunk_size: usize,
}

impl SubsetOps<2> for SubsetOps2d {
    fn copy_to_subdomain<DomainType: DomainView<2>>(
        &self,
        bigger_domain: &DomainType,
        smaller_domain: &mut DomainType,
        threads: usize,
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
        let bigger_min_index =
            bigger_domain.aabb().coord_to_linear(&smaller_min);
        let smaller_min_index = 0;
        let smaller_width = smaller_exclusive_bounds[1] as usize;

        let bigger_bounds = bigger_domain.aabb().bounds;
        let bigger_width =
            (bigger_bounds[(1, 1)] - bigger_bounds[(1, 0)]) as usize + 1;

        let smaller_height = smaller_exclusive_bounds[0] as usize;
        let chunk_size = self.chunk_size.max(smaller_height / threads);
        let chunks = smaller_height.div_ceil(chunk_size);
        //println!("threads: {threads}, chunk_size: {chunk_size}, chunks: {chunks}, smaller_height: {smaller_height}");
        (0..chunks).into_par_iter().for_each(move |c| {
            profiling::scope!("copy_to_subdomain threading callback");

            //println!("THREAD: c: {c}");
            let mut smaller_domain_t = smaller_domain.unsafe_mut_access();
            //let bigger_domain_t = bigger_domain.unsafe_mut_access();
            let mut bigger_min_index_t =
                bigger_min_index + c * chunk_size * bigger_width;
            let mut smaller_min_index_t =
                smaller_min_index + c * chunk_size * smaller_width;
            let start_i = c * chunk_size;
            let end_i = smaller_height.min((c + 1) * chunk_size);
            //println!("THREAD: start_i: {start_i}, end_i: {end_i}");
            for _ in start_i..end_i {
                let smaller_end_index = smaller_min_index_t + smaller_width;
                //println!("THREAD: copy {smaller_min_index_t} -> {smaller_end_index}");
                smaller_domain_t.buffer_mut()
                    [smaller_min_index_t..smaller_end_index]
                    .copy_from_slice(
                        &bigger_domain.buffer()[bigger_min_index_t
                            ..bigger_min_index_t + smaller_width],
                    );
                bigger_min_index_t += bigger_width;
                smaller_min_index_t += smaller_width;
            }
        });
    }

    fn copy_from_subdomain<DomainType: DomainView<2>>(
        &self,
        smaller_domain: &DomainType,
        bigger_domain: &mut DomainType,
        threads: usize,
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
        let bigger_min_index =
            bigger_domain.aabb().coord_to_linear(&smaller_min);
        let smaller_min_index = 0;
        let smaller_width = smaller_exclusive_bounds[1] as usize;

        let bigger_bounds = bigger_domain.aabb().bounds;
        let bigger_width =
            (bigger_bounds[(1, 1)] - bigger_bounds[(1, 0)]) as usize + 1;

        let smaller_height = smaller_exclusive_bounds[0] as usize;
        let chunk_size = self.chunk_size.max(smaller_height / threads);
        let chunks = smaller_height.div_ceil(chunk_size);
        //println!("threads: {threads}, chunk_size: {chunk_size}, chunks: {chunks}, smaller_height: {smaller_height}");
        (0..chunks).into_par_iter().for_each(move |c| {
            profiling::scope!("copy_from_subdomain threading callback");
            //println!("THREAD: c: {c}");
            //let mut smaller_domain_t = smaller_domain.unsafe_mut_access();
            let mut bigger_domain_t = bigger_domain.unsafe_mut_access();
            let mut bigger_min_index_t =
                bigger_min_index + c * chunk_size * bigger_width;
            let mut smaller_min_index_t =
                smaller_min_index + c * chunk_size * smaller_width;
            let start_i = c * chunk_size;
            let end_i = smaller_height.min((c + 1) * chunk_size);
            //println!("THREAD: start_i: {start_i}, end_i: {end_i}");
            for _ in start_i..end_i {
                let smaller_end_index = smaller_min_index_t + smaller_width;
                //println!("THREAD: copy {smaller_min_index_t} -> {smaller_end_index}");
                bigger_domain_t.buffer_mut()
                    [bigger_min_index_t..bigger_min_index_t + smaller_width]
                    .copy_from_slice(
                        &smaller_domain.buffer()
                            [smaller_min_index_t..smaller_end_index],
                    );

                bigger_min_index_t += bigger_width;
                smaller_min_index_t += smaller_width;
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
        ops.copy_to_subdomain(&bigger_domain, &mut smaller_domain, threads);
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
        ops.copy_from_subdomain(&smaller_domain, &mut bigger_domain, threads);
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
