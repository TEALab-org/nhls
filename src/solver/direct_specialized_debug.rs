use crate::domain::*;
use crate::stencil::*;
use crate::util::*;
use std::collections::HashSet;

pub struct AP2DDirectSolverDebug<'a> {
    stencil: &'a Stencil<2, 5>,
    aabb: AABB<2>,
    steps: usize,
    offsets: [i32; 5],
}

impl<'a> AP2DDirectSolverDebug<'a> {
    pub fn new(
        stencil: &'a Stencil<2, 5>,
        aabb: AABB<2>,
        steps: usize,
        _threads: usize,
    ) -> Self {
        let expected_offsets = [
            vector![1, 0],
            vector![0, -1],
            vector![-1, 0],
            vector![0, 1],
            vector![0, 0],
        ];
        debug_assert_eq!(&expected_offsets, stencil.offsets());
        let offsets = aabb.coord_offset_to_linear(stencil.offsets());
        debug_assert_eq!(offsets[4], 0);

        AP2DDirectSolverDebug {
            stencil,
            aabb,
            steps,
            offsets,
        }
    }

    pub fn apply<DomainType: DomainView<2>>(
        &self,
        input: &mut DomainType,
        output: &mut DomainType,
    ) {
        debug_assert_eq!(input.aabb(), output.aabb());
        for _ in 0..self.steps - 1 {
            //global_time += 1;
            self.apply_step(input, output);
            std::mem::swap(input, output);
        }
    }
    pub fn apply_step<DomainType: DomainView<2>>(
        &self,
        input: &mut DomainType,
        output: &mut DomainType,
    ) {
        let w = self.stencil.weights();
        let exclusive_bounds = self.aabb.exclusive_bounds();
        let ib = input.buffer();
        let mut index_set = HashSet::new();

        let x_min = self.aabb.bounds[(0, 0)];
        let x_max = self.aabb.bounds[(0, 1)];
        let y_min = self.aabb.bounds[(1, 0)];
        let y_max = self.aabb.bounds[(0, 1)];

        // Corners
        // (min, min)
        {
            let c = vector![x_min, y_min];
            let e_linear_index = self.aabb.coord_to_linear(&c);
            let linear_index: i32 = 0;
            debug_assert_eq!(e_linear_index, linear_index as usize);
            output.buffer_mut()[linear_index as usize] = w[0]
                * ib[(linear_index + self.offsets[0]) as usize]
                + w[3] * ib[(linear_index + self.offsets[3]) as usize]
                + w[4] * ib[linear_index as usize];
            debug_assert!(!index_set.contains(&linear_index));
            index_set.insert(linear_index);
        }

        // (max, min)
        {
            let c = vector![x_max, y_min];
            let e_linear_index = self.aabb.coord_to_linear(&c);
            let linear_index = (exclusive_bounds[0] - 1) * exclusive_bounds[1];
            debug_assert_eq!(e_linear_index, linear_index as usize);

            output.buffer_mut()[linear_index as usize] = w[2]
                * ib[(linear_index + self.offsets[2]) as usize]
                + w[3] * ib[(linear_index + self.offsets[3]) as usize]
                + w[4] * ib[linear_index as usize];
            debug_assert!(!index_set.contains(&linear_index));
            index_set.insert(linear_index);
        }

        // (min, max)
        {
            let c = vector![x_min, y_max];
            let e_linear_index = self.aabb.coord_to_linear(&c);
            let linear_index = exclusive_bounds[1] - 1;
            debug_assert_eq!(e_linear_index, linear_index as usize);

            output.buffer_mut()[linear_index as usize] = w[0]
                * ib[(linear_index + self.offsets[0]) as usize]
                + w[1] * ib[(linear_index + self.offsets[1]) as usize]
                + w[4] * ib[linear_index as usize];
            debug_assert!(!index_set.contains(&linear_index));
            index_set.insert(linear_index);
        }

        // (max, max)
        {
            let c = vector![x_max, y_max];
            let e_linear_index = self.aabb.coord_to_linear(&c);
            let linear_index = (exclusive_bounds[0] * exclusive_bounds[1]) - 1;
            debug_assert_eq!(e_linear_index, linear_index as usize);

            output.buffer_mut()[linear_index as usize] = w[1]
                * ib[(linear_index + self.offsets[1]) as usize]
                + w[2] * ib[(linear_index + self.offsets[2]) as usize]
                + w[4] * ib[linear_index as usize];
            debug_assert!(!index_set.contains(&linear_index));
            index_set.insert(linear_index);
        }

        // left / right Sides
        for y in 1..exclusive_bounds[1] - 1 {
            // left side
            {
                let c = vector![x_min, y];
                let e_linear_index = self.aabb.coord_to_linear(&c);
                let linear_index = y;
                debug_assert_eq!(e_linear_index, linear_index as usize);

                output.buffer_mut()[linear_index as usize] = w[0]
                    * ib[(linear_index + self.offsets[0]) as usize]
                    + w[1] * ib[(linear_index + self.offsets[1]) as usize]
                    + w[3] * ib[(linear_index + self.offsets[3]) as usize]
                    + w[4] * ib[linear_index as usize];
                debug_assert!(!index_set.contains(&linear_index));
                index_set.insert(linear_index);
            }

            // right side
            {
                let c = vector![x_max, y];
                let e_linear_index = self.aabb.coord_to_linear(&c);
                let linear_index =
                    (exclusive_bounds[0] - 1) * exclusive_bounds[1] + y;
                debug_assert_eq!(e_linear_index, linear_index as usize);

                output.buffer_mut()[linear_index as usize] = w[1]
                    * ib[(linear_index + self.offsets[1]) as usize]
                    + w[2] * ib[(linear_index + self.offsets[2]) as usize]
                    + w[3] * ib[(linear_index + self.offsets[3]) as usize]
                    + w[4] * ib[linear_index as usize];
                debug_assert!(!index_set.contains(&linear_index));
                index_set.insert(linear_index);
            }
        }

        // Central
        for x in 1..exclusive_bounds[0] - 1 {
            let index_base = x * exclusive_bounds[1];

            // top
            {
                let c = vector![x, y_max];
                let e_linear_index = self.aabb.coord_to_linear(&c);
                let linear_index = index_base + exclusive_bounds[1] - 1;
                debug_assert_eq!(e_linear_index, linear_index as usize);

                output.buffer_mut()[linear_index as usize] = w[0]
                    * ib[(linear_index + self.offsets[0]) as usize]
                    + w[1] * ib[(linear_index + self.offsets[1]) as usize]
                    + w[2] * ib[(linear_index + self.offsets[2]) as usize]
                    + w[4] * ib[linear_index as usize];
                debug_assert!(!index_set.contains(&linear_index));
                index_set.insert(linear_index);
            }

            // bottom
            {
                let c = vector![x, y_min];
                let e_linear_index = self.aabb.coord_to_linear(&c);
                let linear_index = index_base;
                debug_assert_eq!(e_linear_index, linear_index as usize);

                output.buffer_mut()[linear_index as usize] = w[0]
                    * ib[(linear_index + self.offsets[0]) as usize]
                    + w[3] * ib[(linear_index + self.offsets[3]) as usize]
                    + w[4] * ib[linear_index as usize];
                debug_assert!(!index_set.contains(&linear_index));
                index_set.insert(linear_index);
            }

            // central
            for y in 1..exclusive_bounds[1] - 1 {
                let c = vector![x, y];
                let e_linear_index = self.aabb.coord_to_linear(&c);
                let linear_index = index_base + y;
                debug_assert_eq!(e_linear_index, linear_index as usize);

                output.buffer_mut()[linear_index as usize] = w[0]
                    * ib[(linear_index + self.offsets[0]) as usize]
                    + w[1] * ib[(linear_index + self.offsets[1]) as usize]
                    + w[2] * ib[(linear_index + self.offsets[2]) as usize]
                    + w[3] * ib[(linear_index + self.offsets[3]) as usize]
                    + w[4] * ib[linear_index as usize];
                debug_assert!(!index_set.contains(&linear_index));
                index_set.insert(linear_index);
            }
        }
    }
}
