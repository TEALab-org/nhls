use crate::domain::*;
use crate::mirror_domain::*;
use crate::stencil::TVStencil;
use crate::util::*;

pub struct SV1DDirectSolver<'a, StencilType: TVStencil<1, 3>> {
    stencil: &'a StencilType,
}

impl<'a, StencilType: TVStencil<1, 3>> SV1DDirectSolver<'a, StencilType> {
    pub fn new(stencil: &'a StencilType) -> Self {
        let expected_offsets = [
            vector![1],  // 0
            vector![-1], // 1
            vector![0],  // 4
        ];
        assert_eq!(&expected_offsets, stencil.offsets());
        SV1DDirectSolver { stencil }
    }

    fn apply_step_double<DomainType: DomainView<1> + Send>(
        &self,
        input_1: &mut DomainType,
        output_1: &mut DomainType,
        input_2: &mut DomainType,
        output_2: &mut DomainType,
        threads: usize,
        global_time: usize,
        n_r: usize,
    ) {
        let w = self.stencil.weights(global_time);
        let ib1 = input_1.buffer();
        let ib2 = input_2.buffer();

        unsafe {
            // Min
            {
                let linear_index: usize = 0;
                *output_1.buffer_mut().get_unchecked_mut(linear_index) =
                    w.get_unchecked(0) * ib1.get_unchecked(linear_index + 1)
                        + w.get_unchecked(1) * ib2.get_unchecked(0)
                        + w.get_unchecked(2) * ib1.get_unchecked(linear_index);
                *output_2.buffer_mut().get_unchecked_mut(linear_index) =
                    w.get_unchecked(0) * ib2.get_unchecked(linear_index + 1)
                        + w.get_unchecked(1) * ib1.get_unchecked(0)
                        + w.get_unchecked(2) * ib2.get_unchecked(linear_index);
            }

            // Max
            {
                let linear_index: usize = n_r - 1;
                *output_1.buffer_mut().get_unchecked_mut(linear_index) =
                    w.get_unchecked(1) * ib1.get_unchecked(linear_index - 1)
                        + w.get_unchecked(2) * ib1.get_unchecked(linear_index);
                *output_2.buffer_mut().get_unchecked_mut(linear_index) =
                    w.get_unchecked(1) * ib2.get_unchecked(linear_index - 1)
                        + w.get_unchecked(2) * ib2.get_unchecked(linear_index);
            }

            let const_output_1: &DomainType = output_1;
            let const_output_2: &DomainType = output_2;
            rayon::scope(|s| {
                let chunk_size = (n_r - 2) / (threads * 2);
                let mut start: usize = 1;
                while start < n_r - 1 {
                    let end = (start + chunk_size).min(n_r - 1);
                    s.spawn(move |_| {
                        let mut o_1 = const_output_1.unsafe_mut_access();
                        let mut o_2 = const_output_2.unsafe_mut_access();

                        for i in start..end {
                            *o_1.buffer_mut().get_unchecked_mut(i) = w
                                .get_unchecked(0)
                                * ib1.get_unchecked(i + 1)
                                + w.get_unchecked(1) * ib1.get_unchecked(i - 1)
                                + w.get_unchecked(2) * ib1.get_unchecked(i);
                            *o_2.buffer_mut().get_unchecked_mut(i) = w
                                .get_unchecked(0)
                                * ib2.get_unchecked(i + 1)
                                + w.get_unchecked(1) * ib2.get_unchecked(i - 1)
                                + w.get_unchecked(2) * ib2.get_unchecked(i);
                        }
                    });
                    start += end;
                }
            });
        }
    }

    fn apply_step_single<DomainType: DomainView<1> + Send>(
        &self,
        input_1: &mut DomainType,
        output_1: &mut DomainType,
        input_2: &mut DomainType,
        output_2: &mut DomainType,
        threads: usize,
        global_time: usize,
        n_r: usize,
    ) {
        let w = self.stencil.weights(global_time);
        let ib1 = input_1.buffer();
        let ib2 = input_2.buffer();

        unsafe {
            // Min
            {
                let linear_index: usize = 0;
                *output_1.buffer_mut().get_unchecked_mut(linear_index) =
                    w.get_unchecked(0) * ib1.get_unchecked(linear_index + 1)
                        + w.get_unchecked(2) * ib1.get_unchecked(linear_index);
                *output_2.buffer_mut().get_unchecked_mut(linear_index) =
                    w.get_unchecked(0) * ib2.get_unchecked(linear_index + 1)
                        + w.get_unchecked(2) * ib2.get_unchecked(linear_index);
            }

            // Max
            {
                let linear_index: usize = n_r - 1;
                *output_1.buffer_mut().get_unchecked_mut(linear_index) =
                    w.get_unchecked(1) * ib1.get_unchecked(linear_index - 1)
                        + w.get_unchecked(2) * ib1.get_unchecked(linear_index);
                *output_2.buffer_mut().get_unchecked_mut(linear_index) =
                    w.get_unchecked(1) * ib2.get_unchecked(linear_index - 1)
                        + w.get_unchecked(2) * ib2.get_unchecked(linear_index);
            }

            let const_output_1: &DomainType = output_1;
            let const_output_2: &DomainType = output_2;
            rayon::scope(|s| {
                let chunk_size = (n_r - 2) / (threads * 2);
                let mut start: usize = 1;
                while start < n_r - 1 {
                    let end = (start + chunk_size).min(n_r - 1);
                    s.spawn(move |_| {
                        let mut o_1 = const_output_1.unsafe_mut_access();
                        let mut o_2 = const_output_2.unsafe_mut_access();

                        for i in start..end {
                            *o_1.buffer_mut().get_unchecked_mut(i) = w
                                .get_unchecked(0)
                                * ib1.get_unchecked(i + 1)
                                + w.get_unchecked(1) * ib1.get_unchecked(i - 1)
                                + w.get_unchecked(2) * ib1.get_unchecked(i);
                            *o_2.buffer_mut().get_unchecked_mut(i) = w
                                .get_unchecked(0)
                                * ib2.get_unchecked(i + 1)
                                + w.get_unchecked(1) * ib2.get_unchecked(i - 1)
                                + w.get_unchecked(2) * ib2.get_unchecked(i);
                        }
                    });
                    start += end;
                }
            });
        }
    }
}

impl<'a, StencilType: TVStencil<1, 3>> SVDirectSolver<1>
    for SV1DDirectSolver<'a, StencilType>
{
    fn apply<'b>(
        &self,
        input_1: &mut SliceDomain<'b, 1>,
        output_1: &mut SliceDomain<'b, 1>,
        input_2: &mut SliceDomain<'b, 1>,
        output_2: &mut SliceDomain<'b, 1>,
        sloped_sides: &Bounds<1>,
        steps: usize,
        mut global_time: usize,
        threads: usize,
    ) {
        debug_assert_eq!(input_1.aabb(), output_2.aabb());

        if *sloped_sides == matrix![0, 1] {
            let n_r = input_1.aabb().buffer_size();
            for _ in 0..steps - 1 {
                self.apply_step_double(
                    input_1,
                    output_1,
                    input_2,
                    output_2,
                    threads,
                    global_time,
                    n_r,
                );
                global_time += 1;
                std::mem::swap(input_1, output_1);
                std::mem::swap(input_2, output_2);
            }
            return;
        }

        let n_r = input_1.aabb().buffer_size();
        for _ in 0..steps - 1 {
            self.apply_step_single(
                input_1,
                output_1,
                input_2,
                output_2,
                threads,
                global_time,
                n_r,
            );
            global_time += 1;
            std::mem::swap(input_1, output_1);
            std::mem::swap(input_2, output_2);
        }
    }
}
