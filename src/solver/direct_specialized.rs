use crate::domain::*;
use crate::stencil::*;
use crate::util::*;

pub struct AP2DDirectSolver<'a> {
    stencil: &'a Stencil<2, 5>,
    aabb: AABB<2>,
    steps: usize,
    threads: usize,
    offsets: [usize; 5],
}

impl<'a> AP2DDirectSolver<'a> {
    pub fn new(
        stencil: &'a Stencil<2, 5>,
        aabb: AABB<2>,
        steps: usize,
        threads: usize,
    ) -> Self {
        let expected_offsets = [
            vector![1, 0],  // 0
            vector![0, -1], // 1
            vector![-1, 0], // 2
            vector![0, 1],  // 3
            vector![0, 0],  // 4
        ];
        debug_assert_eq!(&expected_offsets, stencil.offsets());
        let offsets_i32 = aabb.coord_offset_to_linear(stencil.offsets());
        let offsets = [
            offsets_i32[0].unsigned_abs() as usize,
            offsets_i32[1].unsigned_abs() as usize,
            offsets_i32[2].unsigned_abs() as usize,
            offsets_i32[3].unsigned_abs() as usize,
            offsets_i32[4].unsigned_abs() as usize,
        ];
        println!("offsets: {:?}", offsets);

        AP2DDirectSolver {
            stencil,
            aabb,
            steps,
            threads,
            offsets,
        }
    }

    pub fn apply<DomainType: DomainView<2> + Send>(
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
    pub fn apply_step<DomainType: DomainView<2> + Send>(
        &self,
        input: &mut DomainType,
        output: &mut DomainType,
    ) {
        let w = self.stencil.weights();
        let exclusive_bounds = self.aabb.exclusive_bounds();
        let ib = input.buffer();
        let const_output: &DomainType = output;
        unsafe {
            rayon::scope(|s| {
                s.spawn(move |_| {
                    let mut o = const_output.unsafe_mut_access();
                    // Corners
                    // (min, min)
                    {
                        let linear_index: usize = 0;
                        *o.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(0)
                            * ib.get_unchecked(
                                linear_index + self.offsets.get_unchecked(0),
                            )
                            + w.get_unchecked(3)
                                * ib.get_unchecked(
                                    linear_index
                                        + self.offsets.get_unchecked(3),
                                )
                            + w.get_unchecked(4)
                                * ib.get_unchecked(linear_index);
                    }

                    // (max, min)
                    {
                        let linear_index: usize =
                            ((exclusive_bounds.get_unchecked(0) - 1)
                                * exclusive_bounds.get_unchecked(1))
                                as usize;

                        *o.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(2)
                            * ib.get_unchecked(
                                linear_index - self.offsets.get_unchecked(2),
                            )
                            + w.get_unchecked(3)
                                * ib.get_unchecked(
                                    linear_index
                                        + self.offsets.get_unchecked(3),
                                )
                            + w.get_unchecked(4)
                                * ib.get_unchecked(linear_index);
                    }

                    // (min, max)
                    {
                        let linear_index: usize =
                            (exclusive_bounds.get_unchecked(1) - 1) as usize;

                        *o.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(0)
                            * ib.get_unchecked(
                                linear_index + self.offsets.get_unchecked(0),
                            )
                            + w.get_unchecked(1)
                                * ib.get_unchecked(
                                    linear_index
                                        - self.offsets.get_unchecked(1),
                                )
                            + w.get_unchecked(4)
                                * ib.get_unchecked(linear_index);
                    }

                    // (max, max)
                    {
                        let linear_index: usize =
                            ((exclusive_bounds.get_unchecked(0)
                                * exclusive_bounds.get_unchecked(1))
                                - 1) as usize;

                        *o.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(1)
                            * ib.get_unchecked(
                                linear_index - self.offsets.get_unchecked(1),
                            )
                            + w.get_unchecked(2)
                                * ib.get_unchecked(
                                    linear_index
                                        - self.offsets.get_unchecked(2),
                                )
                            + w.get_unchecked(4)
                                * ib.get_unchecked(linear_index);
                    }

                    // left / right Sides
                    for y in 1..(exclusive_bounds.get_unchecked(1) - 1) as usize
                    {
                        // left side
                        {
                            let linear_index: usize = y;

                            *o.buffer_mut().get_unchecked_mut(linear_index) = w
                                .get_unchecked(0)
                                * ib.get_unchecked(
                                    linear_index
                                        + self.offsets.get_unchecked(0),
                                )
                                + w.get_unchecked(1)
                                    * ib.get_unchecked(
                                        linear_index
                                            - self.offsets.get_unchecked(1),
                                    )
                                + w.get_unchecked(3)
                                    * ib.get_unchecked(
                                        linear_index
                                            + self.offsets.get_unchecked(3),
                                    )
                                + w.get_unchecked(4)
                                    * ib.get_unchecked(linear_index);
                        }

                        // right side
                        {
                            let linear_index: usize =
                                ((exclusive_bounds.get_unchecked(0) - 1)
                                    * exclusive_bounds.get_unchecked(1)
                                    + y as i32)
                                    as usize;

                            *o.buffer_mut().get_unchecked_mut(linear_index) = w
                                .get_unchecked(1)
                                * ib.get_unchecked(
                                    linear_index
                                        - self.offsets.get_unchecked(1),
                                )
                                + w.get_unchecked(2)
                                    * ib.get_unchecked(
                                        linear_index
                                            - self.offsets.get_unchecked(2),
                                    )
                                + w.get_unchecked(3)
                                    * ib.get_unchecked(
                                        linear_index
                                            + self.offsets.get_unchecked(3),
                                    )
                                + w.get_unchecked(4)
                                    * ib.get_unchecked(linear_index);
                        }
                    }
                });

                let chunk_size = (*exclusive_bounds.get_unchecked(0) as usize
                    - 2)
                    / (self.threads * 2);
                let mut start: usize = 1;
                while start < (exclusive_bounds.get_unchecked(0) - 1) as usize {
                    let end = (start + chunk_size)
                        .min(*exclusive_bounds.get_unchecked(0) as usize - 1);
                    s.spawn(move |_| {
                        let mut o = const_output.unsafe_mut_access();

                        // Central (on x axis)
                        for x in start..end {
                            let index_base =
                                x * *exclusive_bounds.get_unchecked(1) as usize;

                            // top
                            {
                                let linear_index: usize = index_base
                                    + *exclusive_bounds.get_unchecked(1)
                                        as usize
                                    - 1;
                                *o.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib.get_unchecked(
                                        linear_index
                                            + self.offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(1)
                                        * ib.get_unchecked(
                                            linear_index
                                                - self.offsets.get_unchecked(1),
                                        )
                                    + w.get_unchecked(2)
                                        * ib.get_unchecked(
                                            linear_index
                                                - self.offsets.get_unchecked(2),
                                        )
                                    + w.get_unchecked(4)
                                        * ib.get_unchecked(linear_index);
                            }

                            // bottom
                            {
                                let linear_index: usize = index_base;
                                *o.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib.get_unchecked(
                                        linear_index
                                            + self.offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(3)
                                        * ib.get_unchecked(
                                            linear_index
                                                + self.offsets.get_unchecked(3),
                                        )
                                    + w.get_unchecked(4)
                                        * ib.get_unchecked(linear_index);
                            }

                            // central
                            for y in 1..*exclusive_bounds.get_unchecked(1)
                                as usize
                                - 1
                            {
                                let linear_index: usize = index_base + y;
                                *o.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib.get_unchecked(
                                        linear_index
                                            + self.offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(1)
                                        * ib.get_unchecked(
                                            linear_index
                                                - self.offsets.get_unchecked(1),
                                        )
                                    + w.get_unchecked(2)
                                        * ib.get_unchecked(
                                            linear_index
                                                - self.offsets.get_unchecked(2),
                                        )
                                    + w.get_unchecked(3)
                                        * ib.get_unchecked(
                                            linear_index
                                                + self.offsets.get_unchecked(3),
                                        )
                                    + w.get_unchecked(4)
                                        * ib.get_unchecked(linear_index);
                            }
                        }
                    });
                    start += chunk_size;
                }
            });
        }
    }
}
