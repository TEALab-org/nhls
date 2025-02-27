use crate::domain::*;
use crate::par_stencil;
use crate::stencil::*;
use crate::util::*;
use std::collections::HashSet;

pub struct AP2DDirectSolverDebug<'a> {
    stencil: &'a Stencil<2, 5>,
    aabb: AABB<2>,
    steps: usize,
    threads: usize,
    offsets: [i32; 5],
}

impl<'a> AP2DDirectSolverDebug<'a> {
    pub fn new(
        stencil: &'a Stencil<2, 5>,
        aabb: AABB<2>,
        steps: usize,
        threads: usize,
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
            threads,
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

pub fn box_apply<
    BC,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    DomainType: DomainView<GRID_DIMENSION>,
>(
    bc: &BC,
    stencil: &Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    input: &mut DomainType,
    output: &mut DomainType,
    steps: usize,
    mut global_time: usize,
    chunk_size: usize,
) where
    BC: BCCheck<GRID_DIMENSION>,
{
    debug_assert_eq!(input.aabb(), output.aabb());
    for _ in 0..steps - 1 {
        global_time += 1;
        par_stencil::apply(bc, stencil, input, output, global_time, chunk_size);
        std::mem::swap(input, output);
    }

    // Run central solve with linear offsets
    // Triple for loop?
    // Calculate the linear offset for both buffers as we go

    // so we need custum gather args like construct
    // that we can update over time that includes all

    // Run boundary solve

    global_time += 1;
    par_stencil::apply(bc, stencil, input, output, global_time, chunk_size);
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::util::*;
    use fftw::array::AlignedVec;
    use float_cmp::assert_approx_eq;
    use nalgebra::matrix;

    fn test_unit_stencil<
        BC,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    >(
        stencil: &Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        bc_lookup: &BC,
        bound: &AABB<GRID_DIMENSION>,
        steps: usize,
    ) where
        BC: BCCheck<GRID_DIMENSION>,
    {
        let chunk_size = 3;
        assert_approx_eq!(f64, stencil.apply(&Values::from_element(1.0)), 1.0);

        let mut input_domain = OwnedDomain::new(*bound);
        let mut output_domain = OwnedDomain::new(*bound);

        input_domain.par_set_values(|_| 1.0, chunk_size);

        box_apply(
            bc_lookup,
            stencil,
            &mut input_domain,
            &mut output_domain,
            steps,
            0,
            chunk_size,
        );

        for x in output_domain.buffer() {
            assert_approx_eq!(f64, *x, 1.0);
        }
    }

    #[test]
    fn test_1d_simple() {
        let stencil = Stencil::new([[0]], |args: &[f64; 1]| args[0]);
        let max_size = AABB::new(matrix![0, 99]);
        let lookup = ConstantCheck::new(1.0, max_size);
        test_unit_stencil(&stencil, &lookup, &max_size, 100);
    }

    #[test]
    fn test_2d_simple() {
        let stencil = Stencil::new([[0, 0]], |args: &[f64; 1]| args[0]);
        let max_size = AABB::new(matrix![0, 49; 0, 49]);
        let lookup = ConstantCheck::new(1.0, max_size);
        test_unit_stencil(&stencil, &lookup, &max_size, 9);
    }

    #[test]
    fn test_2d_less_simple() {
        let stencil = Stencil::new(
            [[0, -1], [0, 1], [1, 0], [-1, 0], [0, 0]],
            |args: &[f64; 5]| {
                let mut r = 0.0;
                for a in args {
                    r += a / 5.0;
                }
                r
            },
        );
        let max_size = AABB::new(matrix![0, 49; 0, 49]);
        let lookup = ConstantCheck::new(1.0, max_size);
        test_unit_stencil(&stencil, &lookup, &max_size, 10);
    }

    #[test]
    fn test_1d_less_simple() {
        let stencil = Stencil::new([[-1], [1], [0]], |args: &[f64; 3]| {
            debug_assert_eq!(args.len(), 3);
            let mut r = 0.0;
            for a in args {
                r += a / 3.0
            }
            r
        });
        let max_size = AABB::new(matrix![0, 99]);
        let lookup = ConstantCheck::new(1.0, max_size);
        test_unit_stencil(&stencil, &lookup, &max_size, 10);
    }

    #[test]
    fn test_3d() {
        let stencil = Stencil::new(
            [
                [0, 0, -2],
                [4, 5, 3],
                [0, -1, 0],
                [0, 1, 0],
                [1, 0, 0],
                [-1, 0, 4],
                [0, 0, 0],
            ],
            |args: &[f64; 7]| {
                let mut r = 0.0;
                for a in args {
                    r += a / 7.0;
                }
                r
            },
        );
        {
            let max_size = AABB::new(matrix![0, 19; 0, 19; 0, 19]);
            let lookup = ConstantCheck::new(1.0, max_size);
            test_unit_stencil(&stencil, &lookup, &max_size, 5);
        }

        {
            let max_size = AABB::new(matrix![0, 10; 0, 8; 0, 19]);
            let lookup = ConstantCheck::new(1.0, max_size);
            test_unit_stencil(&stencil, &lookup, &max_size, 5);
        }
    }

    #[test]
    fn shifter() {
        let chunk_size = 1;
        let stencil = Stencil::new([[-1]], |args: &[f64; 1]| args[0]);
        let max_size = AABB::new(matrix![0, 9]);
        let mut input_buffer = AlignedVec::new(10);
        for i in 0..10 {
            input_buffer[i] = i as f64;
        }

        let mut input_domain = OwnedDomain::new(max_size);
        let mut output_domain = OwnedDomain::new(max_size);

        input_domain
            .par_set_values(|coord: Coord<1>| coord[0] as f64, chunk_size);

        let bc_lookup = ConstantCheck::new(-1.0, max_size);
        let steps = 3;

        box_apply(
            &bc_lookup,
            &stencil,
            &mut input_domain,
            &mut output_domain,
            steps,
            0,
            chunk_size,
        );
        for i in 0..3 {
            assert_approx_eq!(f64, output_domain.buffer()[i], -1.0);
        }
        for i in 3..10 {
            assert_approx_eq!(f64, output_domain.buffer()[i], (i - 3) as f64);
        }
    }
}
