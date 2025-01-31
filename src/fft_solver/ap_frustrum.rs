use crate::fft_solver::*;
use crate::util::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Side {
    Min,
    Max,
}

impl std::fmt::Display for Side {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
        match self {
            Side::Min => write!(f, "Min"),
            Side::Max => write!(f, "Max"),
        }
    }
}

impl Side {
    #[inline]
    pub fn inner_index(&self) -> usize {
        match self {
            Side::Min => 1,
            Side::Max => 0,
        }
    }

    #[inline]
    pub fn outer_index(&self) -> usize {
        match self {
            Side::Min => 0,
            Side::Max => 1,
        }
    }

    #[inline]
    fn inner_coef(&self) -> i32 {
        match self {
            Side::Min => -1,
            Side::Max => 1,
        }
    }

    #[inline]
    fn outer_coef(&self) -> i32 {
        match self {
            Side::Min => 1,
            Side::Max => -1,
        }
    }
}

// Given some output aabb
#[derive(Debug, PartialEq, Eq)]
pub struct APFrustrum<const GRID_DIMENSION: usize> {
    pub output_aabb: AABB<GRID_DIMENSION>,
    pub recursion_dimension: usize,
    pub side: Side,
    pub steps: usize,
}

impl<const GRID_DIMENSION: usize> APFrustrum<GRID_DIMENSION> {
    pub fn new(
        output_aabb: AABB<GRID_DIMENSION>,
        recursion_dimension: usize,
        side: Side,
        steps: usize,
    ) -> Self {
        APFrustrum {
            output_aabb,
            recursion_dimension,
            side,
            steps,
        }
    }

    pub fn sloped_sides(&self) -> Bounds<GRID_DIMENSION> {
        let mut result = Bounds::from_element(1);
        result[(self.recursion_dimension, self.side.outer_index())] = 0;
        for d in self.recursion_dimension + 1..GRID_DIMENSION {
            result[(d, 0)] = 0;
            result[(d, 1)] = 0;
        }
        result
    }

    pub fn input_aabb(
        &self,
        stencil_slopes: &Bounds<GRID_DIMENSION>,
    ) -> AABB<GRID_DIMENSION> {
        let sloped_sides = self.sloped_sides();
        frustrum_input_aabb(
            self.steps,
            &self.output_aabb,
            &sloped_sides,
            stencil_slopes,
        )
    }

    // Steps from input face,
    // slice off the rest, return it
    pub fn time_cut(
        &mut self,
        cut_steps: usize,
        stencil_slopes: &Bounds<GRID_DIMENSION>,
    ) -> Option<APFrustrum<GRID_DIMENSION>> {
        // The input_aabb for this frustrum doesn't change...
        // but we define frustrum by output_aabb
        // so we need to calculate that
        // the output
        debug_assert!(cut_steps <= self.steps);
        // No time cut
        if cut_steps >= self.steps {
            return None;
        }

        let remaining_steps = self.steps - cut_steps;
        let next_frustrum = APFrustrum::new(
            self.output_aabb,
            self.recursion_dimension,
            self.side,
            remaining_steps,
        );
        self.output_aabb = next_frustrum.input_aabb(stencil_slopes);
        self.steps = cut_steps;
        //println!("timecut: {}", cut_steps);
        Some(next_frustrum)
    }

    /// complement to decompose
    pub fn periodic_solve_output(
        &self,
        stencil_slopes: &Bounds<GRID_DIMENSION>,
    ) -> AABB<GRID_DIMENSION> {
        // sloped sides are part of periodic solve
        // so we want to 1-0 flip
        let boundary_sides = flip_sloped(&self.sloped_sides());
        self.output_aabb;
        self.output_aabb
    }

    // TODO: add tests
    // in particular,
    pub fn decompose(&self) -> Vec<APFrustrum<GRID_DIMENSION>> {
        // Cause FFT goes steps in, so sub one, shrug
        // !!!! danger, was -1  here, uh oh
        let i_steps = self.steps as i32 - 1;
        let mut result = Vec::new();

        // 1 for this dimension,
        let mut outer_aabb = self.output_aabb;
        let outer_bound = self.output_aabb.bounds
            [(self.recursion_dimension, self.side.outer_index())];
        outer_aabb.bounds
            // TODO: Stencil slope here!
            [(self.recursion_dimension, self.side.inner_index())] =
            outer_bound + self.side.outer_coef() * i_steps;
        result.push(APFrustrum::new(
            outer_aabb,
            self.recursion_dimension,
            self.side,
            self.steps,
        ));

        let mut remainder = self.output_aabb;
        remainder.bounds
            [(self.recursion_dimension, self.side.outer_index())] +=
            self.side.outer_coef() * i_steps;

        // 2 for each lower dimension
        for d in self.recursion_dimension + 1..GRID_DIMENSION {
            let mut min_aabb = remainder;
            let min_bound = min_aabb.bounds[(d, 0)];
            min_aabb.bounds[(d, 1)] = min_bound + i_steps;
            result.push(APFrustrum::new(min_aabb, d, Side::Min, self.steps));

            let mut max_aabb = remainder;
            let max_bound = max_aabb.bounds[(d, 1)];
            max_aabb.bounds[(d, Side::Max.inner_index())] = max_bound - i_steps;
            result.push(APFrustrum::new(max_aabb, d, Side::Max, self.steps));
        }

        result
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn decompose() {
        // 1D
        {
            let aabb = AABB::new(matrix![0, 10]);
            println!("aabb: {:?}", aabb);
            let f1 = APFrustrum::new(aabb, 0, Side::Min, 2);
            let d1 = f1.decompose();
            assert_eq!(d1.len(), 1);
            assert_eq!(
                d1[0],
                APFrustrum::new(AABB::new(matrix![0, 1]), 0, Side::Min, 2)
            );

            let f2 = APFrustrum::new(aabb, 0, Side::Max, 2);
            let d2 = f2.decompose();
            assert_eq!(d2.len(), 1);
            assert_eq!(
                d2[0],
                APFrustrum::new(AABB::new(matrix![9, 10]), 0, Side::Max, 2)
            );
        }

        // 2D d0
        {
            let steps = 20;
            let aabb = AABB::new(matrix![0, 50; 0, 200]);
            let f1 = APFrustrum::new(aabb, 0, Side::Min, steps);
            let d1 = f1.decompose();
            assert_eq!(d1.len(), 3);
            assert_eq!(
                d1[0],
                APFrustrum::new(
                    AABB::new(matrix![0, 19; 0, 200]),
                    0,
                    Side::Min,
                    steps,
                )
            );
            assert_eq!(
                d1[1],
                APFrustrum::new(
                    AABB::new(matrix![19, 50; 0, 19]),
                    1,
                    Side::Min,
                    steps,
                )
            );
            assert_eq!(
                d1[2],
                APFrustrum::new(
                    AABB::new(matrix![19, 50; 181, 200]),
                    1,
                    Side::Max,
                    steps,
                )
            );

            let f2 = APFrustrum::new(aabb, 0, Side::Max, steps);
            let d2 = f2.decompose();
            assert_eq!(d2.len(), 3);
            assert_eq!(
                d2[0],
                APFrustrum::new(
                    AABB::new(matrix![31, 50; 0, 200]),
                    0,
                    Side::Max,
                    steps,
                )
            );
            assert_eq!(
                d2[1],
                APFrustrum::new(
                    AABB::new(matrix![0, 31; 0, 19]),
                    1,
                    Side::Min,
                    steps
                )
            );
            assert_eq!(
                d2[2],
                APFrustrum::new(
                    AABB::new(matrix![0, 31; 181, 200]),
                    1,
                    Side::Max,
                    steps,
                )
            );
        }

        // 2D d1
        {
            let steps = 20;
            let aabb = AABB::new(matrix![0, 200; 0, 50]);
            let f1 = APFrustrum::new(aabb, 1, Side::Min, steps);
            let d1 = f1.decompose();
            assert_eq!(d1.len(), 1);
            assert_eq!(
                d1[0],
                APFrustrum::new(
                    AABB::new(matrix![0, 200; 0, 19]),
                    1,
                    Side::Min,
                    steps,
                )
            );

            let f2 = APFrustrum::new(aabb, 1, Side::Max, steps);
            let d2 = f2.decompose();
            assert_eq!(d2.len(), 1);
            assert_eq!(
                d2[0],
                APFrustrum::new(
                    AABB::new(matrix![0, 200; 31, 50]),
                    1,
                    Side::Max,
                    steps,
                )
            );
        }
    }

    #[test]
    fn sloped_sides_test() {
        {
            let output_aabb = AABB::new(matrix![20, 40; 20, 40]);
            let f = APFrustrum::new(output_aabb, 0, Side::Min, 10);
            debug_assert_eq!(f.sloped_sides(), matrix![0, 1; 0, 0]);
        }

        {
            let output_aabb = AABB::new(matrix![20, 40; 20, 40]);
            let f = APFrustrum::new(output_aabb, 0, Side::Max, 10);
            debug_assert_eq!(f.sloped_sides(), matrix![1, 0; 0, 0]);
        }

        {
            let output_aabb = AABB::new(matrix![20, 40; 20, 40]);
            let f = APFrustrum::new(output_aabb, 1, Side::Min, 10);
            debug_assert_eq!(f.sloped_sides(), matrix![1, 1; 0, 1]);
        }

        {
            let output_aabb = AABB::new(matrix![20, 40; 20, 40]);
            let f = APFrustrum::new(output_aabb, 1, Side::Max, 10);
            debug_assert_eq!(f.sloped_sides(), matrix![1, 1; 1, 0]);
        }
    }

    #[test]
    fn time_cut_test() {
        // Cut greater than or equal is no cut at all
        {
            let ss = matrix![1, 1; 1, 1];
            let output_aabb = AABB::new(matrix![20, 40; 20, 40]);
            let mut f = APFrustrum::new(output_aabb, 1, Side::Max, 10);
            //debug_assert_eq!(f.time_cut(11, &ss), None);
        }

        {
            let ss = matrix![1, 1; 1, 1];
            let output_aabb = AABB::new(matrix![20, 40; 20, 40]);
            let mut f = APFrustrum::new(output_aabb, 1, Side::Max, 10);
            debug_assert_eq!(f.time_cut(10, &ss), None);
        }

        // Cut 1 to start
        {
            let ss = matrix![1, 1; 1, 1];
            let output_aabb = AABB::new(matrix![20, 40; 20, 40]);
            let mut f = APFrustrum::new(output_aabb, 1, Side::Max, 10);

            //let expected_output_aabb =
            //   debug_assert_eq!(f.time_cut(1, &ss), None);
        }
    }

    /*
        // Test that decomp + center = bounds exactly
        // by checking every coordinate appears once.
        fn test_decomp<const DIMENSION: usize>(
            frustrum: &APFrustrum<DIMENSION>,
        ) {
            let mut coord_set = std::collections::HashSet::new();
            coord_set.extend(center.coord_iter());

            let d = bounds.decomposition(center);
            for [b1, b2] in d {
                for c in b1.coord_iter().chain(b2.coord_iter()) {
                    assert!(!coord_set.contains(&c));
                    coord_set.insert(c);
                }
            }

            for c in bounds.coord_iter() {
                assert!(coord_set.contains(&c));
            }

            println!("{}", coord_set.len());
        }
    */
}
