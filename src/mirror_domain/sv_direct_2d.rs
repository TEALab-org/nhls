use crate::domain::*;
use crate::mirror_domain::*;
use crate::stencil::TVStencil;
use crate::util::*;

pub struct SV2DDirectSolver<'a, StencilType: TVStencil<2, 5>> {
    stencil: &'a StencilType,
}

impl<'a, StencilType: TVStencil<2, 5>> SV2DDirectSolver<'a, StencilType> {
    pub fn new(stencil: &'a StencilType) -> Self {
        let expected_offsets = [
            vector![1, 0],  // 0
            vector![0, -1], // 1
            vector![-1, 0], // 2
            vector![0, 1],  // 3
            vector![0, 0],  // 4
        ];
        assert_eq!(&expected_offsets, stencil.offsets());
        SV2DDirectSolver { stencil }
    }

    fn apply_step_double<DomainType: DomainView<2> + Send>(
        &self,
        input_1: &mut DomainType,
        output_1: &mut DomainType,
        input_2: &mut DomainType,
        output_2: &mut DomainType,
        threads: usize,
        global_time: usize,
        offsets: [usize; 5],
        exclusive_bounds: Coord<2>,
    ) {
        let w = self.stencil.weights(global_time);
        let ib1 = input_1.buffer();
        let ib2 = input_2.buffer();

        let const_output_1: &DomainType = output_1;
        let const_output_2: &DomainType = output_2;
        unsafe {
            rayon::scope(|s| {
                s.spawn(move |_| {
                    let mut o1 = const_output_1.unsafe_mut_access();
                    let mut o2 = const_output_2.unsafe_mut_access();

                    // Corners
                    // (min, min)
                    {
                        let linear_index: usize = 0;
                        *o1.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(0)
                            * ib1.get_unchecked(
                                linear_index + offsets.get_unchecked(0),
                            )
                            + w.get_unchecked(2)
                                * ib2.get_unchecked(linear_index)
                            + w.get_unchecked(3)
                                * ib1.get_unchecked(
                                    linear_index + offsets.get_unchecked(3),
                                )
                            + w.get_unchecked(4)
                                * ib1.get_unchecked(linear_index);

                        *o2.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(0)
                            * ib2.get_unchecked(
                                linear_index + offsets.get_unchecked(0),
                            )
                            + w.get_unchecked(2)
                                * ib1.get_unchecked(linear_index)
                            + w.get_unchecked(3)
                                * ib2.get_unchecked(
                                    linear_index + offsets.get_unchecked(3),
                                )
                            + w.get_unchecked(4)
                                * ib2.get_unchecked(linear_index);
                    }

                    // (max, min)
                    {
                        let linear_index: usize =
                            ((exclusive_bounds.get_unchecked(0) - 1)
                                * exclusive_bounds.get_unchecked(1))
                                as usize;

                        *o1.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(2)
                            * ib1.get_unchecked(
                                linear_index - offsets.get_unchecked(2),
                            )
                            + w.get_unchecked(3)
                                * ib1.get_unchecked(
                                    linear_index + offsets.get_unchecked(3),
                                )
                            + w.get_unchecked(4)
                                * ib1.get_unchecked(linear_index);

                        *o2.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(2)
                            * ib2.get_unchecked(
                                linear_index - offsets.get_unchecked(2),
                            )
                            + w.get_unchecked(3)
                                * ib2.get_unchecked(
                                    linear_index + offsets.get_unchecked(3),
                                )
                            + w.get_unchecked(4)
                                * ib2.get_unchecked(linear_index);
                    }

                    // (min, max)
                    {
                        let linear_index: usize =
                            (exclusive_bounds.get_unchecked(1) - 1) as usize;

                        *o1.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(0)
                            * ib1.get_unchecked(
                                linear_index + offsets.get_unchecked(0),
                            )
                            + w.get_unchecked(1)
                                * ib1.get_unchecked(
                                    linear_index - offsets.get_unchecked(1),
                                )
                            + w.get_unchecked(4)
                                * ib1.get_unchecked(linear_index);

                        *o2.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(0)
                            * ib2.get_unchecked(
                                linear_index + offsets.get_unchecked(0),
                            )
                            + w.get_unchecked(1)
                                * ib2.get_unchecked(
                                    linear_index - offsets.get_unchecked(1),
                                )
                            + w.get_unchecked(4)
                                * ib2.get_unchecked(linear_index);
                    }

                    // (max, max)
                    {
                        let linear_index: usize =
                            ((exclusive_bounds.get_unchecked(0)
                                * exclusive_bounds.get_unchecked(1))
                                - 1) as usize;

                        *o1.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(1)
                            * ib1.get_unchecked(
                                linear_index - offsets.get_unchecked(1),
                            )
                            + w.get_unchecked(2)
                                * ib1.get_unchecked(
                                    linear_index - offsets.get_unchecked(2),
                                )
                            + w.get_unchecked(4)
                                * ib1.get_unchecked(linear_index);

                        *o2.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(1)
                            * ib2.get_unchecked(
                                linear_index - offsets.get_unchecked(1),
                            )
                            + w.get_unchecked(2)
                                * ib2.get_unchecked(
                                    linear_index - offsets.get_unchecked(2),
                                )
                            + w.get_unchecked(4)
                                * ib2.get_unchecked(linear_index);
                    }

                    // left / right Sides
                    for y in 1..(exclusive_bounds.get_unchecked(1) - 1) as usize
                    {
                        // left side
                        {
                            let linear_index: usize = y;

                            *o1.buffer_mut().get_unchecked_mut(linear_index) = w
                                .get_unchecked(0)
                                * ib1.get_unchecked(
                                    linear_index + offsets.get_unchecked(0),
                                )
                                + w.get_unchecked(1)
                                    * ib1.get_unchecked(
                                        linear_index - offsets.get_unchecked(1),
                                    )
                                + w.get_unchecked(2)
                                    * ib2.get_unchecked(linear_index)
                                + w.get_unchecked(3)
                                    * ib1.get_unchecked(
                                        linear_index + offsets.get_unchecked(3),
                                    )
                                + w.get_unchecked(4)
                                    * ib1.get_unchecked(linear_index);

                            *o2.buffer_mut().get_unchecked_mut(linear_index) = w
                                .get_unchecked(0)
                                * ib2.get_unchecked(
                                    linear_index + offsets.get_unchecked(0),
                                )
                                + w.get_unchecked(1)
                                    * ib2.get_unchecked(
                                        linear_index - offsets.get_unchecked(1),
                                    )
                                + w.get_unchecked(3)
                                    * ib2.get_unchecked(
                                        linear_index + offsets.get_unchecked(3),
                                    )
                                + w.get_unchecked(4)
                                    * ib2.get_unchecked(linear_index);
                        }

                        // right side
                        {
                            let linear_index: usize =
                                ((exclusive_bounds.get_unchecked(0) - 1)
                                    * exclusive_bounds.get_unchecked(1)
                                    + y as i32)
                                    as usize;

                            *o1.buffer_mut().get_unchecked_mut(linear_index) = w
                                .get_unchecked(1)
                                * ib1.get_unchecked(
                                    linear_index - offsets.get_unchecked(1),
                                )
                                + w.get_unchecked(2)
                                    * ib1.get_unchecked(
                                        linear_index - offsets.get_unchecked(2),
                                    )
                                + w.get_unchecked(3)
                                    * ib1.get_unchecked(
                                        linear_index + offsets.get_unchecked(3),
                                    )
                                + w.get_unchecked(4)
                                    * ib1.get_unchecked(linear_index);

                            *o2.buffer_mut().get_unchecked_mut(linear_index) = w
                                .get_unchecked(1)
                                * ib2.get_unchecked(
                                    linear_index - offsets.get_unchecked(1),
                                )
                                + w.get_unchecked(2)
                                    * ib2.get_unchecked(
                                        linear_index - offsets.get_unchecked(2),
                                    )
                                + w.get_unchecked(3)
                                    * ib2.get_unchecked(
                                        linear_index + offsets.get_unchecked(3),
                                    )
                                + w.get_unchecked(4)
                                    * ib2.get_unchecked(linear_index);
                        }
                    }
                });

                let chunk_size = (*exclusive_bounds.get_unchecked(0) as usize
                    - 2)
                    / (threads * 2);
                let mut start: usize = 1;
                while start < (exclusive_bounds.get_unchecked(0) - 1) as usize {
                    let end = (start + chunk_size)
                        .min(*exclusive_bounds.get_unchecked(0) as usize - 1);
                    s.spawn(move |_| {
                        let mut o1 = const_output_1.unsafe_mut_access();
                        let mut o2 = const_output_2.unsafe_mut_access();

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
                                *o1.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib1.get_unchecked(
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(1)
                                        * ib1.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(1),
                                        )
                                    + w.get_unchecked(2)
                                        * ib1.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(2),
                                        )
                                    + w.get_unchecked(4)
                                        * ib1.get_unchecked(linear_index);

                                *o2.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib2.get_unchecked(
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(1)
                                        * ib2.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(1),
                                        )
                                    + w.get_unchecked(2)
                                        * ib2.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(2),
                                        )
                                    + w.get_unchecked(4)
                                        * ib2.get_unchecked(linear_index);
                            }

                            // bottom
                            {
                                let linear_index: usize = index_base;
                                *o1.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib1.get_unchecked(
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(3)
                                        * ib1.get_unchecked(
                                            linear_index
                                                + offsets.get_unchecked(3),
                                        )
                                    + w.get_unchecked(4)
                                        * ib1.get_unchecked(linear_index);

                                *o2.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib2.get_unchecked(
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(3)
                                        * ib2.get_unchecked(
                                            linear_index
                                                + offsets.get_unchecked(3),
                                        )
                                    + w.get_unchecked(4)
                                        * ib2.get_unchecked(linear_index);
                            }

                            // central
                            for y in 1..*exclusive_bounds.get_unchecked(1)
                                as usize
                                - 1
                            {
                                let linear_index: usize = index_base + y;
                                *o2.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib2.get_unchecked(
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(1)
                                        * ib2.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(1),
                                        )
                                    + w.get_unchecked(2)
                                        * ib2.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(2),
                                        )
                                    + w.get_unchecked(3)
                                        * ib2.get_unchecked(
                                            linear_index
                                                + offsets.get_unchecked(3),
                                        )
                                    + w.get_unchecked(4)
                                        * ib2.get_unchecked(linear_index);

                                *o2.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib2.get_unchecked(
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(1)
                                        * ib2.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(1),
                                        )
                                    + w.get_unchecked(2)
                                        * ib2.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(2),
                                        )
                                    + w.get_unchecked(3)
                                        * ib2.get_unchecked(
                                            linear_index
                                                + offsets.get_unchecked(3),
                                        )
                                    + w.get_unchecked(4)
                                        * ib2.get_unchecked(linear_index);
                            }
                        }
                    });
                    start += chunk_size;
                }
            });
        }
    }

    fn apply_step_single<DomainType: DomainView<2> + Send>(
        &self,
        input_1: &mut DomainType,
        output_1: &mut DomainType,
        input_2: &mut DomainType,
        output_2: &mut DomainType,
        threads: usize,
        global_time: usize,
        offsets: [usize; 5],
        exclusive_bounds: Coord<2>,
    ) {
        let w = self.stencil.weights(global_time);
        let ib1 = input_1.buffer();
        let ib2 = input_2.buffer();

        let const_output_1: &DomainType = output_1;
        let const_output_2: &DomainType = output_2;
        unsafe {
            rayon::scope(|s| {
                s.spawn(move |_| {
                    let mut o1 = const_output_1.unsafe_mut_access();
                    let mut o2 = const_output_2.unsafe_mut_access();

                    // Corners
                    // (min, min)
                    {
                        let linear_index: usize = 0;
                        *o1.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(0)
                            * ib1.get_unchecked(
                                linear_index + offsets.get_unchecked(0),
                            )
                            + w.get_unchecked(3)
                                * ib1.get_unchecked(
                                    linear_index + offsets.get_unchecked(3),
                                )
                            + w.get_unchecked(4)
                                * ib1.get_unchecked(linear_index);

                        *o2.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(0)
                            * ib2.get_unchecked(
                                linear_index + offsets.get_unchecked(0),
                            )
                            + w.get_unchecked(3)
                                * ib2.get_unchecked(
                                    linear_index + offsets.get_unchecked(3),
                                )
                            + w.get_unchecked(4)
                                * ib2.get_unchecked(linear_index);
                    }

                    // (max, min)
                    {
                        let linear_index: usize =
                            ((exclusive_bounds.get_unchecked(0) - 1)
                                * exclusive_bounds.get_unchecked(1))
                                as usize;

                        *o1.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(2)
                            * ib1.get_unchecked(
                                linear_index - offsets.get_unchecked(2),
                            )
                            + w.get_unchecked(3)
                                * ib1.get_unchecked(
                                    linear_index + offsets.get_unchecked(3),
                                )
                            + w.get_unchecked(4)
                                * ib1.get_unchecked(linear_index);

                        *o2.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(2)
                            * ib2.get_unchecked(
                                linear_index - offsets.get_unchecked(2),
                            )
                            + w.get_unchecked(3)
                                * ib2.get_unchecked(
                                    linear_index + offsets.get_unchecked(3),
                                )
                            + w.get_unchecked(4)
                                * ib2.get_unchecked(linear_index);
                    }

                    // (min, max)
                    {
                        let linear_index: usize =
                            (exclusive_bounds.get_unchecked(1) - 1) as usize;

                        *o1.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(0)
                            * ib1.get_unchecked(
                                linear_index + offsets.get_unchecked(0),
                            )
                            + w.get_unchecked(1)
                                * ib1.get_unchecked(
                                    linear_index - offsets.get_unchecked(1),
                                )
                            + w.get_unchecked(4)
                                * ib1.get_unchecked(linear_index);

                        *o2.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(0)
                            * ib2.get_unchecked(
                                linear_index + offsets.get_unchecked(0),
                            )
                            + w.get_unchecked(1)
                                * ib2.get_unchecked(
                                    linear_index - offsets.get_unchecked(1),
                                )
                            + w.get_unchecked(4)
                                * ib2.get_unchecked(linear_index);
                    }

                    // (max, max)
                    {
                        let linear_index: usize =
                            ((exclusive_bounds.get_unchecked(0)
                                * exclusive_bounds.get_unchecked(1))
                                - 1) as usize;

                        *o1.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(1)
                            * ib1.get_unchecked(
                                linear_index - offsets.get_unchecked(1),
                            )
                            + w.get_unchecked(2)
                                * ib1.get_unchecked(
                                    linear_index - offsets.get_unchecked(2),
                                )
                            + w.get_unchecked(4)
                                * ib1.get_unchecked(linear_index);

                        *o2.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(1)
                            * ib2.get_unchecked(
                                linear_index - offsets.get_unchecked(1),
                            )
                            + w.get_unchecked(2)
                                * ib2.get_unchecked(
                                    linear_index - offsets.get_unchecked(2),
                                )
                            + w.get_unchecked(4)
                                * ib2.get_unchecked(linear_index);
                    }

                    // left / right Sides
                    for y in 1..(exclusive_bounds.get_unchecked(1) - 1) as usize
                    {
                        // left side
                        {
                            let linear_index: usize = y;

                            *o1.buffer_mut().get_unchecked_mut(linear_index) = w
                                .get_unchecked(0)
                                * ib1.get_unchecked(
                                    linear_index + offsets.get_unchecked(0),
                                )
                                + w.get_unchecked(1)
                                    * ib1.get_unchecked(
                                        linear_index - offsets.get_unchecked(1),
                                    )
                                + w.get_unchecked(3)
                                    * ib1.get_unchecked(
                                        linear_index + offsets.get_unchecked(3),
                                    )
                                + w.get_unchecked(4)
                                    * ib1.get_unchecked(linear_index);

                            *o2.buffer_mut().get_unchecked_mut(linear_index) = w
                                .get_unchecked(0)
                                * ib2.get_unchecked(
                                    linear_index + offsets.get_unchecked(0),
                                )
                                + w.get_unchecked(1)
                                    * ib2.get_unchecked(
                                        linear_index - offsets.get_unchecked(1),
                                    )
                                + w.get_unchecked(3)
                                    * ib2.get_unchecked(
                                        linear_index + offsets.get_unchecked(3),
                                    )
                                + w.get_unchecked(4)
                                    * ib2.get_unchecked(linear_index);
                        }

                        // right side
                        {
                            let linear_index: usize =
                                ((exclusive_bounds.get_unchecked(0) - 1)
                                    * exclusive_bounds.get_unchecked(1)
                                    + y as i32)
                                    as usize;

                            *o1.buffer_mut().get_unchecked_mut(linear_index) = w
                                .get_unchecked(1)
                                * ib1.get_unchecked(
                                    linear_index - offsets.get_unchecked(1),
                                )
                                + w.get_unchecked(2)
                                    * ib1.get_unchecked(
                                        linear_index - offsets.get_unchecked(2),
                                    )
                                + w.get_unchecked(3)
                                    * ib1.get_unchecked(
                                        linear_index + offsets.get_unchecked(3),
                                    )
                                + w.get_unchecked(4)
                                    * ib1.get_unchecked(linear_index);

                            *o2.buffer_mut().get_unchecked_mut(linear_index) = w
                                .get_unchecked(1)
                                * ib2.get_unchecked(
                                    linear_index - offsets.get_unchecked(1),
                                )
                                + w.get_unchecked(2)
                                    * ib2.get_unchecked(
                                        linear_index - offsets.get_unchecked(2),
                                    )
                                + w.get_unchecked(3)
                                    * ib2.get_unchecked(
                                        linear_index + offsets.get_unchecked(3),
                                    )
                                + w.get_unchecked(4)
                                    * ib2.get_unchecked(linear_index);
                        }
                    }
                });

                let chunk_size = (*exclusive_bounds.get_unchecked(0) as usize
                    - 2)
                    / (threads * 2);
                let mut start: usize = 1;
                while start < (exclusive_bounds.get_unchecked(0) - 1) as usize {
                    let end = (start + chunk_size)
                        .min(*exclusive_bounds.get_unchecked(0) as usize - 1);
                    s.spawn(move |_| {
                        let mut o1 = const_output_1.unsafe_mut_access();
                        let mut o2 = const_output_2.unsafe_mut_access();

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
                                *o1.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib1.get_unchecked(
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(1)
                                        * ib1.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(1),
                                        )
                                    + w.get_unchecked(2)
                                        * ib1.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(2),
                                        )
                                    + w.get_unchecked(4)
                                        * ib1.get_unchecked(linear_index);

                                *o2.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib2.get_unchecked(
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(1)
                                        * ib2.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(1),
                                        )
                                    + w.get_unchecked(2)
                                        * ib2.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(2),
                                        )
                                    + w.get_unchecked(4)
                                        * ib2.get_unchecked(linear_index);
                            }

                            // bottom
                            {
                                let linear_index: usize = index_base;
                                *o1.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib1.get_unchecked(
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(3)
                                        * ib1.get_unchecked(
                                            linear_index
                                                + offsets.get_unchecked(3),
                                        )
                                    + w.get_unchecked(4)
                                        * ib1.get_unchecked(linear_index);

                                *o2.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib2.get_unchecked(
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(3)
                                        * ib2.get_unchecked(
                                            linear_index
                                                + offsets.get_unchecked(3),
                                        )
                                    + w.get_unchecked(4)
                                        * ib2.get_unchecked(linear_index);
                            }

                            // central
                            for y in 1..*exclusive_bounds.get_unchecked(1)
                                as usize
                                - 1
                            {
                                let linear_index: usize = index_base + y;
                                *o2.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib2.get_unchecked(
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(1)
                                        * ib2.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(1),
                                        )
                                    + w.get_unchecked(2)
                                        * ib2.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(2),
                                        )
                                    + w.get_unchecked(3)
                                        * ib2.get_unchecked(
                                            linear_index
                                                + offsets.get_unchecked(3),
                                        )
                                    + w.get_unchecked(4)
                                        * ib2.get_unchecked(linear_index);

                                *o2.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib2.get_unchecked(
                                        linear_index + offsets.get_unchecked(0),
                                    )
                                    + w.get_unchecked(1)
                                        * ib2.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(1),
                                        )
                                    + w.get_unchecked(2)
                                        * ib2.get_unchecked(
                                            linear_index
                                                - offsets.get_unchecked(2),
                                        )
                                    + w.get_unchecked(3)
                                        * ib2.get_unchecked(
                                            linear_index
                                                + offsets.get_unchecked(3),
                                        )
                                    + w.get_unchecked(4)
                                        * ib2.get_unchecked(linear_index);
                            }
                        }
                    });
                    start += chunk_size;
                }
            });
        }
    }
}

impl<'a, StencilType: TVStencil<2, 5>> SVDirectSolver<2>
    for SV2DDirectSolver<'a, StencilType>
{
    fn apply<'b>(
        &self,
        input_1: &mut SliceDomain<'b, 2>,
        output_1: &mut SliceDomain<'b, 2>,
        input_2: &mut SliceDomain<'b, 2>,
        output_2: &mut SliceDomain<'b, 2>,
        sloped_sides: &Bounds<2>,
        steps: usize,
        mut global_time: usize,
        threads: usize,
    ) {
        debug_assert_eq!(input_1.aabb(), output_2.aabb());
        let offsets_i32 = input_1
            .aabb()
            .coord_offset_to_linear(self.stencil.offsets());
        let offsets = [
            offsets_i32[0].unsigned_abs() as usize,
            offsets_i32[1].unsigned_abs() as usize,
            offsets_i32[2].unsigned_abs() as usize,
            offsets_i32[3].unsigned_abs() as usize,
            offsets_i32[4].unsigned_abs() as usize,
        ];
        let exclusive_bounds = input_1.aabb().exclusive_bounds();

        if *sloped_sides == matrix![0, 1; 0, 0] {
            for _ in 0..steps - 1 {
                self.apply_step_double(
                    input_1,
                    output_1,
                    input_2,
                    output_2,
                    threads,
                    global_time,
                    offsets,
                    exclusive_bounds,
                );
                global_time += 1;
                std::mem::swap(input_1, output_1);
                std::mem::swap(input_2, output_2);
            }
            return;
        }

        for _ in 0..steps - 1 {
            self.apply_step_single(
                input_1,
                output_1,
                input_2,
                output_2,
                threads,
                global_time,
                offsets,
                exclusive_bounds,
            );
            global_time += 1;
            std::mem::swap(input_1, output_1);
            std::mem::swap(input_2, output_2);
        }
    }
}
