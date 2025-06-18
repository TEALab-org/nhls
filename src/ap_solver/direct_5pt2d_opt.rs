use crate::ap_solver::direct_solver::DirectSolver;
use crate::domain::*;
use crate::stencil::TVStencil;
use crate::util::*;

pub struct DirectSolver5Pt2DOpt<'a, StencilType: TVStencil<2, 5>> {
    stencil: &'a StencilType,
}

impl<'a, StencilType: TVStencil<2, 5>> DirectSolver5Pt2DOpt<'a, StencilType> {
    pub fn new(stencil: &'a StencilType) -> Self {
        let expected_offsets = [
            vector![1, 0],  // 0
            vector![0, -1], // 1
            vector![-1, 0], // 2
            vector![0, 1],  // 3
            vector![0, 0],  // 4
        ];
        assert_eq!(&expected_offsets, stencil.offsets());
        DirectSolver5Pt2DOpt { stencil }
    }

    fn apply_step<DomainType: DomainView<2> + Send>(
        &self,
        input: &mut DomainType,
        output: &mut DomainType,
        threads: usize,
        global_time: usize,
        offsets: [usize; 5],
        exclusive_bounds: Coord<2>,
    ) {
        let w = self.stencil.weights(global_time);
        let ib = input.buffer();
        let const_output: &DomainType = output;
        unsafe {
            rayon::scope(|s| {
                s.spawn(move |_| {
                    profiling::scope!("direct_solver: special Thread Callback");
                    let mut o = const_output.unsafe_mut_access();
                    // Corners
                    // (min, min)
                    {
                        let linear_index: usize = 0;
                        *o.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(0)
                            * ib.get_unchecked(
                                linear_index + offsets.get_unchecked(0),
                            )
                            + w.get_unchecked(3)
                                * ib.get_unchecked(
                                    linear_index + offsets.get_unchecked(3),
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
                                linear_index - offsets.get_unchecked(2),
                            )
                            + w.get_unchecked(3)
                                * ib.get_unchecked(
                                    linear_index + offsets.get_unchecked(3),
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
                                linear_index + offsets.get_unchecked(0),
                            )
                            + w.get_unchecked(1)
                                * ib.get_unchecked(
                                    linear_index - offsets.get_unchecked(1),
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
                                linear_index - offsets.get_unchecked(1),
                            )
                            + w.get_unchecked(2)
                                * ib.get_unchecked(
                                    linear_index - offsets.get_unchecked(2),
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
                                    linear_index + offsets.get_unchecked(0),
                                )
                                + w.get_unchecked(1)
                                    * ib.get_unchecked(
                                        linear_index - offsets.get_unchecked(1),
                                    )
                                + w.get_unchecked(3)
                                    * ib.get_unchecked(
                                        linear_index + offsets.get_unchecked(3),
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
                                    linear_index - offsets.get_unchecked(1),
                                )
                                + w.get_unchecked(2)
                                    * ib.get_unchecked(
                                        linear_index - offsets.get_unchecked(2),
                                    )
                                + w.get_unchecked(3)
                                    * ib.get_unchecked(
                                        linear_index + offsets.get_unchecked(3),
                                    )
                                + w.get_unchecked(4)
                                    * ib.get_unchecked(linear_index);
                        }
                    }
                });

                let chunk_size =
                    (*exclusive_bounds.get_unchecked(0) as usize - 2) / threads;
                let mut start: usize = 1;
                while start < (exclusive_bounds.get_unchecked(0) - 1) as usize {
                    let end = (start + chunk_size)
                        .min(*exclusive_bounds.get_unchecked(0) as usize - 1);
                    s.spawn(move |_| {
                        profiling::scope!("direct_solver: Thread Callback");
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
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(1)
                                        * ib.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(1),
                                        )
                                    + w.get_unchecked(2)
                                        * ib.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(2),
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
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(3)
                                        * ib.get_unchecked(
                                            linear_index
                                                + offsets.get_unchecked(3),
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
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(1)
                                        * ib.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(1),
                                        )
                                    + w.get_unchecked(2)
                                        * ib.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(2),
                                        )
                                    + w.get_unchecked(3)
                                        * ib.get_unchecked(
                                            linear_index
                                                + offsets.get_unchecked(3),
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

impl<StencilType: TVStencil<2, 5>> DirectSolver<2>
    for DirectSolver5Pt2DOpt<'_, StencilType>
{
    fn apply<'b>(
        &self,
        input: &mut SliceDomain<'b, 2>,
        output: &mut SliceDomain<'b, 2>,
        _sloped_sides: &Bounds<2>,
        steps: usize,
        mut global_time: usize,
        threads: usize,
    ) {
        profiling::scope!("direct_solver");
        debug_assert_eq!(input.aabb(), output.aabb());

        let offsets_i32 =
            input.aabb().coord_offset_to_linear(self.stencil.offsets());
        let offsets = [
            offsets_i32[0].unsigned_abs() as usize,
            offsets_i32[1].unsigned_abs() as usize,
            offsets_i32[2].unsigned_abs() as usize,
            offsets_i32[3].unsigned_abs() as usize,
            offsets_i32[4].unsigned_abs() as usize,
        ];

        let exclusive_bounds = input.aabb().exclusive_bounds();
        for _ in 0..steps - 1 {
            self.apply_step(
                input,
                output,
                threads,
                global_time,
                offsets,
                exclusive_bounds,
            );
            global_time += 1;
            std::mem::swap(input, output);
        }
    }
}
