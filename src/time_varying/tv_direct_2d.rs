use crate::domain::*;
use crate::time_varying::*;
use crate::util::*;
use rayon::prelude::*;
use std::collections::HashSet;

// Used to direct solve frustrum regions.
pub struct TVDirectFrustrumSolver2D<'a, StencilType: TVStencil<2, 5>> {
    pub stencil: &'a StencilType,
    pub stencil_slopes: Bounds<2>,
}

impl<'a, StencilType: TVStencil<2, 5>> TVDirectFrustrumSolver2D<'a, StencilType> {
    pub fn new(stencil: &'a StencilType) -> Self {
        let stencil_slopes = matrix![1, 1; 1, 1];
        let expected_offsets = [
            vector![1, 0],  // 0
            vector![0, -1], // 1
            vector![-1, 0], // 2
            vector![0, 1],  // 3
            vector![0, 0],  // 4
        ];
        debug_assert_eq!(&expected_offsets, stencil.offsets());

        TVDirectFrustrumSolver2D {
            stencil,
            stencil_slopes,
        }
    }
    pub fn apply_x_min_step<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, 2>,
        output_domain: &mut SliceDomain<'b, 2>,
        global_time: usize,
    ) {
        let input_exclusive_bounds = input_domain.aabb().exclusive_bounds();
        let output_exclusive_bounds = output_domain.aabb().exclusive_bounds();

        let ib = input_domain.buffer();
        let offsets_i32 = input_domain
            .aabb()
            .coord_offset_to_linear(self.stencil.offsets());
        let offsets = [
            offsets_i32[0].unsigned_abs() as usize,
            offsets_i32[1].unsigned_abs() as usize,
            offsets_i32[2].unsigned_abs() as usize,
            offsets_i32[3].unsigned_abs() as usize,
            offsets_i32[4].unsigned_abs() as usize,
        ];
        let w = self.stencil.weights(global_time);

        let const_output: &SliceDomain<'b, 2> = output_domain;
        unsafe {
            rayon::scope(|s| {
                s.spawn(move |_| {
                    let mut o = const_output.unsafe_mut_access();
                    // Corners
                    // (min, min)
                    {
                        let input_linear_index: usize = 0;
                        let output_linear_index: usize = 0;
                        *o.buffer_mut()
                            .get_unchecked_mut(output_linear_index) = w
                            .get_unchecked(0)
                            * ib.get_unchecked(
                                input_linear_index + offsets.get_unchecked(0),
                            )
                            + w.get_unchecked(3)
                                * ib.get_unchecked(
                                    input_linear_index
                                        + offsets.get_unchecked(3),
                                )
                            + w.get_unchecked(4)
                                * ib.get_unchecked(input_linear_index);
                    }

                    // (min, max)
                    {
                        let input_linear_index: usize =
                            (input_exclusive_bounds.get_unchecked(1) - 1)
                                as usize;
                        let output_linear_index: usize =
                            (output_exclusive_bounds.get_unchecked(1) - 1)
                                as usize;

                        *o.buffer_mut()
                            .get_unchecked_mut(output_linear_index) = w
                            .get_unchecked(0)
                            * ib.get_unchecked(
                                input_linear_index + offsets.get_unchecked(0),
                            )
                            + w.get_unchecked(1)
                                * ib.get_unchecked(
                                    input_linear_index
                                        - offsets.get_unchecked(1),
                                )
                            + w.get_unchecked(4)
                                * ib.get_unchecked(input_linear_index);
                    }

                    // left sides
                    for y in 1..(input_exclusive_bounds.get_unchecked(1) - 1)
                        as usize
                    {
                        // left side
                        {
                            let input_linear_index: usize = y;
                            let output_linear_index: usize = y;

                            *o.buffer_mut()
                                .get_unchecked_mut(output_linear_index) = w
                                .get_unchecked(0)
                                * ib.get_unchecked(
                                    input_linear_index
                                        + offsets.get_unchecked(0),
                                )
                                + w.get_unchecked(1)
                                    * ib.get_unchecked(
                                        input_linear_index
                                            - offsets.get_unchecked(1),
                                    )
                                + w.get_unchecked(3)
                                    * ib.get_unchecked(
                                        input_linear_index
                                            + offsets.get_unchecked(3),
                                    )
                                + w.get_unchecked(4)
                                    * ib.get_unchecked(input_linear_index);
                        }
                    }
                });

                let chunk_size =
                    (*output_exclusive_bounds.get_unchecked(0) as usize - 1)
                        / (2);
                let mut start: usize = 1;
                while start
                    < (*output_exclusive_bounds.get_unchecked(0)) as usize
                {
                    let end = (start + chunk_size).min(
                        *output_exclusive_bounds.get_unchecked(0) as usize - 1,
                    );
                    s.spawn(move |_| {
                        let mut o = const_output.unsafe_mut_access();

                        // Central (on x axis)
                        for x in start..end {
                            let index_base = x * *output_exclusive_bounds
                                .get_unchecked(1)
                                as usize;

                            // top
                            {
                                let linear_index: usize = index_base
                                    + *output_exclusive_bounds.get_unchecked(1)
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
                            for y in 1..*output_exclusive_bounds
                                .get_unchecked(1)
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

    pub fn apply_x_max_step<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, 2>,
        output_domain: &mut SliceDomain<'b, 2>,
        global_time: usize,
    ) {
        let input_exclusive_bounds = input_domain.aabb().exclusive_bounds();
        let output_exclusive_bounds = output_domain.aabb().exclusive_bounds();

        let ib = input_domain.buffer();
        let offsets_i32 = input_domain
            .aabb()
            .coord_offset_to_linear(self.stencil.offsets());
        let offsets = [
            offsets_i32[0].unsigned_abs() as usize,
            offsets_i32[1].unsigned_abs() as usize,
            offsets_i32[2].unsigned_abs() as usize,
            offsets_i32[3].unsigned_abs() as usize,
            offsets_i32[4].unsigned_abs() as usize,
        ];
        let w = self.stencil.weights(global_time);

        let const_output: &SliceDomain<'b, 2> = output_domain;
        unsafe {
            rayon::scope(|s| {
                let mut o = const_output.unsafe_mut_access();
                {
                    let linear_index: usize =
                        ((output_exclusive_bounds.get_unchecked(0) - 1)
                            * output_exclusive_bounds.get_unchecked(1))
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
                        + w.get_unchecked(4) * ib.get_unchecked(linear_index);
                }

                // (max, max)
                {
                    let linear_index: usize =
                        ((output_exclusive_bounds.get_unchecked(0)
                            * output_exclusive_bounds.get_unchecked(1))
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
                        + w.get_unchecked(4) * ib.get_unchecked(linear_index);
                }

                // left / right Sides
                for y in 1..(output_exclusive_bounds.get_unchecked(1) - 1) as usize {
                    // right side
                    {
                        let linear_index: usize =
                            ((output_exclusive_bounds.get_unchecked(0) - 1)
                                * output_exclusive_bounds.get_unchecked(1)
                                + y as i32)
                                as usize;

                        *o.buffer_mut().get_unchecked_mut(linear_index) = w
                            .get_unchecked(1)
                            * ib.get_unchecked(
                                linear_index - offsets.get_unchecked(1),
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

                let chunk_size = (*output_exclusive_bounds.get_unchecked(0) as usize
                    - 2)
                    / (2);
                let mut start: usize = 0;
                while start < (output_exclusive_bounds.get_unchecked(0)  -1 ) as usize {
                    let end = (start + chunk_size)
                        .min(*output_exclusive_bounds.get_unchecked(0) as usize - 1);
                    s.spawn(move |_| {
                        let mut o = const_output.unsafe_mut_access();

                        // Central (on x axis)
                        for x in start..end {
                            let index_base =
                                x * *output_exclusive_bounds.get_unchecked(1) as usize;

                            // top
                            {
                                let linear_index: usize = index_base
                                    + *output_exclusive_bounds.get_unchecked(1)
                                        as usize
                                    - 1;
                                *o.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib.get_unchecked(
                                        linear_index
                                            + offsets.get_unchecked(0),
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
                                        linear_index
                                            + offsets.get_unchecked(0),
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
                            for y in 1..*output_exclusive_bounds.get_unchecked(1)
                                as usize
                                - 1
                            {
                                let linear_index: usize = index_base + y;
                                *o.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib.get_unchecked(
                                        linear_index
                                            + offsets.get_unchecked(0),
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

    pub fn apply_y_min_step<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, 2>,
        output_domain: &mut SliceDomain<'b, 2>,
        global_time: usize,
    ) {
        let input_exclusive_bounds = input_domain.aabb().exclusive_bounds();
        let output_exclusive_bounds = output_domain.aabb().exclusive_bounds();

        let ib = input_domain.buffer();
        let offsets_i32 = input_domain
            .aabb()
            .coord_offset_to_linear(self.stencil.offsets());
        let offsets = [
            offsets_i32[0].unsigned_abs() as usize,
            offsets_i32[1].unsigned_abs() as usize,
            offsets_i32[2].unsigned_abs() as usize,
            offsets_i32[3].unsigned_abs() as usize,
            offsets_i32[4].unsigned_abs() as usize,
        ];
        let w = self.stencil.weights(global_time);

        let const_output: &SliceDomain<'b, 2> = output_domain;
        unsafe {
            rayon::scope(|s| {
                let chunk_size = (*output_exclusive_bounds.get_unchecked(0) as usize)
                    / (2);
                let mut start: usize = 0;
                while start < (*output_exclusive_bounds.get_unchecked(0)) as usize {
                    let end = (start + chunk_size)
                        .min(*output_exclusive_bounds.get_unchecked(0) as usize - 1);
                    s.spawn(move |_| {
                        let mut o = const_output.unsafe_mut_access();

                        // Central (on x axis)
                        for x in start..end {
                            let index_base =
                                x * *output_exclusive_bounds.get_unchecked(1) as usize;

                            {
                                let linear_index: usize = index_base;
                                *o.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib.get_unchecked(
                                        linear_index
                                            + offsets.get_unchecked(0),
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
                            for y in 1..*output_exclusive_bounds.get_unchecked(1)
                                as usize
                                - 1
                            {
                                let linear_index: usize = index_base + y;
                                *o.buffer_mut()
                                    .get_unchecked_mut(linear_index) = w
                                    .get_unchecked(0)
                                    * ib.get_unchecked(
                                        linear_index
                                            + offsets.get_unchecked(0),
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

    pub fn apply_y_max_step<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, 2>,
        output_domain: &mut SliceDomain<'b, 2>,
        global_time: usize,
    ) {
        let input_exclusive_bounds = input_domain.aabb().exclusive_bounds();
        let output_exclusive_bounds = output_domain.aabb().exclusive_bounds();

        let ib = input_domain.buffer();
        let offsets_i32 = input_domain
            .aabb()
            .coord_offset_to_linear(self.stencil.offsets());
        let offsets = [
            offsets_i32[0].unsigned_abs() as usize,
            offsets_i32[1].unsigned_abs() as usize,
            offsets_i32[2].unsigned_abs() as usize,
            offsets_i32[3].unsigned_abs() as usize,
            offsets_i32[4].unsigned_abs() as usize,
        ];
        let w = self.stencil.weights(global_time);

        let const_output: &SliceDomain<'b, 2> = output_domain;
        unsafe {
            rayon::scope(|s| {});
        }
    }

    pub fn apply_x_min<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, 2>,
        output_domain: &mut SliceDomain<'b, 2>,
        steps: usize,
        mut global_time: usize,
    ) {
        let x_min: Bounds<2> = matrix![0, 1; 0, 0];
        let mut trapezoid_slopes = self.stencil_slopes.component_mul(&x_min);
        let negative_slopes = -1 * trapezoid_slopes.column(1);
        trapezoid_slopes.set_column(1, &negative_slopes);

        let mut output_box = *input_domain.aabb();
        for _ in 0..steps {
            global_time += 1;
            output_box = output_box.add_bounds_diff(trapezoid_slopes);
            debug_assert!(
                input_domain.aabb().buffer_size() >= output_box.buffer_size()
            );
            output_domain.set_aabb(output_box);

            self.apply_x_min_step(
                input_domain,
                output_domain,
                global_time,
            );

            std::mem::swap(input_domain, output_domain);
        }
        std::mem::swap(input_domain, output_domain);
    }

    pub fn apply_x_max<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, 2>,
        output_domain: &mut SliceDomain<'b, 2>,
        steps: usize,
        mut global_time: usize,
    ) {
        let x_max: Bounds<2> = matrix![1, 0; 0, 0];
        let mut trapezoid_slopes = self.stencil_slopes.component_mul(&x_max);
        let negative_slopes = -1 * trapezoid_slopes.column(1);
        trapezoid_slopes.set_column(1, &negative_slopes);

        let mut output_box = *input_domain.aabb();
        for _ in 0..steps {
            global_time += 1;
            output_box = output_box.add_bounds_diff(trapezoid_slopes);
            debug_assert!(
                input_domain.aabb().buffer_size() >= output_box.buffer_size()
            );
            output_domain.set_aabb(output_box);

            self.apply_x_max_step(
                input_domain,
                output_domain,
                global_time,
            );

            std::mem::swap(input_domain, output_domain);
        }
        std::mem::swap(input_domain, output_domain);
    }

    pub fn apply_y_min<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, 2>,
        output_domain: &mut SliceDomain<'b, 2>,
        steps: usize,
        mut global_time: usize,
    ) {
        let y_min: Bounds<2> = matrix![1, 1; 0, 1];
        let mut trapezoid_slopes = self.stencil_slopes.component_mul(&y_min);
        let negative_slopes = -1 * trapezoid_slopes.column(1);
        trapezoid_slopes.set_column(1, &negative_slopes);

        let mut output_box = *input_domain.aabb();
        for _ in 0..steps {
            global_time += 1;
            output_box = output_box.add_bounds_diff(trapezoid_slopes);
            debug_assert!(
                input_domain.aabb().buffer_size() >= output_box.buffer_size()
            );
            output_domain.set_aabb(output_box);

            self.apply_y_min_step(
                input_domain,
                output_domain,
                global_time,
            );

            std::mem::swap(input_domain, output_domain);
        }
        std::mem::swap(input_domain, output_domain);
    }

    pub fn apply_y_max<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, 2>,
        output_domain: &mut SliceDomain<'b, 2>,
        steps: usize,
        mut global_time: usize,
    ) {
        let y_max: Bounds<2> = matrix![1, 1; 1, 0];
        let mut trapezoid_slopes = self.stencil_slopes.component_mul(&y_max);
        let negative_slopes = -1 * trapezoid_slopes.column(1);
        trapezoid_slopes.set_column(1, &negative_slopes);

        let mut output_box = *input_domain.aabb();
        for _ in 0..steps {
            global_time += 1;
            output_box = output_box.add_bounds_diff(trapezoid_slopes);
            debug_assert!(
                input_domain.aabb().buffer_size() >= output_box.buffer_size()
            );
            output_domain.set_aabb(output_box);

            self.apply_y_max_step(
                input_domain,
                output_domain,
                global_time,
            );

            std::mem::swap(input_domain, output_domain);
        }
        std::mem::swap(input_domain, output_domain);
    }

    pub fn apply<'b>(
        &self,
        input_domain: &mut SliceDomain<'b, 2>,
        output_domain: &mut SliceDomain<'b, 2>,
        sloped_sides: &Bounds<2>,
        steps: usize,
        mut global_time: usize,
    ) {
        assert_eq!(input_domain.aabb(), output_domain.aabb());

        let x_min: Bounds<2> = matrix![0, 1; 0, 0];
        let x_max: Bounds<2> = matrix![1, 0; 0, 0];
        let y_min: Bounds<2> = matrix![1, 1; 0, 1];
        let y_max: Bounds<2> = matrix![1, 1; 1, 0];

        match sloped_sides {
            s if x_min == *s => {
                self.apply_x_min(
                    input_domain,
                    output_domain,
                    steps,
                    global_time,
                );
            }
            s if x_max == *s => {
                self.apply_x_max(
                    input_domain,
                    output_domain,
                    steps,
                    global_time,
                );
            }
            s if y_min == *s => {
                self.apply_y_min(
                    input_domain,
                    output_domain,
                    steps,
                    global_time,
                );
            }
            s if y_max == *s => {
                self.apply_y_max(
                    input_domain,
                    output_domain,
                    steps,
                    global_time,
                );
            }
            s => {
                panic!("Unknown sloped_sides: {:?}", s);
            }
        }

        let mut trapezoid_slopes =
            self.stencil_slopes.component_mul(sloped_sides);
        let negative_slopes = -1 * trapezoid_slopes.column(1);
        trapezoid_slopes.set_column(1, &negative_slopes);

        let mut output_box = *input_domain.aabb();
        for _ in 0..steps {
            global_time += 1;
            output_box = output_box.add_bounds_diff(trapezoid_slopes);
            debug_assert!(
                input_domain.aabb().buffer_size() >= output_box.buffer_size()
            );
            output_domain.set_aabb(output_box);

            // SOLVE

            std::mem::swap(input_domain, output_domain);
        }
        std::mem::swap(input_domain, output_domain);
    }
}
