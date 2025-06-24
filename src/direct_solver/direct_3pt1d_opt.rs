use crate::direct_solver::*;
use crate::domain::*;
use crate::stencil::TVStencil;
use crate::util::*;
use crate::SolverInterface;

/// Optimized direct solver for 3pt 1D stencil.
/// Implements a constant zero boundary condition.
pub struct DirectSolver3Pt1DOpt<'a, StencilType: TVStencil<1, 3>> {
    stencil: &'a StencilType,
}

impl<'a, StencilType: TVStencil<1, 3>> DirectSolver3Pt1DOpt<'a, StencilType> {
    pub fn new(stencil: &'a StencilType) -> Self {
        let expected_offsets = [
            vector![1],  // 0
            vector![-1], // 1
            vector![0],  // 4
        ];
        assert_eq!(&expected_offsets, stencil.offsets());
        DirectSolver3Pt1DOpt { stencil }
    }

    fn apply_step<DomainType: DomainView<1> + Send>(
        &self,
        input: &mut DomainType,
        output: &mut DomainType,
        threads: usize,
        global_time: usize,
        n_r: usize,
    ) {
        let w = self.stencil.weights(global_time);
        let ib = input.buffer();

        unsafe {
            // Min
            {
                let linear_index: usize = 0;
                *output.buffer_mut().get_unchecked_mut(linear_index) =
                    w.get_unchecked(0) * ib.get_unchecked(linear_index + 1)
                        + w.get_unchecked(2) * ib.get_unchecked(linear_index);
            }

            // Max
            {
                let linear_index: usize = n_r - 1;
                *output.buffer_mut().get_unchecked_mut(linear_index) =
                    w.get_unchecked(1) * ib.get_unchecked(linear_index - 1)
                        + w.get_unchecked(2) * ib.get_unchecked(linear_index);
            }

            let const_output: &DomainType = output;
            rayon::scope(|s| {
                profiling::scope!("direct_solver: Thread Callback");
                let chunk_size = (n_r - 2) / (threads * 2);
                let mut start: usize = 1;
                while start < n_r - 1 {
                    let end = (start + chunk_size).min(n_r - 1);
                    s.spawn(move |_| {
                        let mut o = const_output.unsafe_mut_access();
                        for i in start..end {
                            *o.buffer_mut().get_unchecked_mut(i) = w
                                .get_unchecked(0)
                                * ib.get_unchecked(i + 1)
                                + w.get_unchecked(1) * ib.get_unchecked(i - 1)
                                + w.get_unchecked(2) * ib.get_unchecked(i);
                        }
                    });
                    start += chunk_size;
                }
            });
        }
    }
}

impl<StencilType: TVStencil<1, 3>> DirectSolverInterface<1>
    for DirectSolver3Pt1DOpt<'_, StencilType>
{
    fn apply<'b>(
        &self,
        input: &mut SliceDomain<'b, 1>,
        output: &mut SliceDomain<'b, 1>,
        _sloped_sides: &Bounds<1>,
        steps: usize,
        mut global_time: usize,
        threads: usize,
    ) {
        debug_assert_eq!(input.aabb(), output.aabb());

        let n_r = input.aabb().buffer_size();
        for _ in 0..steps - 1 {
            self.apply_step(input, output, threads, global_time, n_r);
            global_time += 1;
            std::mem::swap(input, output);
        }
        std::mem::swap(input, output);
    }
}

pub struct Direct3Pt1DSolver<'a, StencilType: TVStencil<1, 3>> {
    solver: DirectSolver3Pt1DOpt<'a, StencilType>,
    steps: usize,
    threads: usize,
}

impl<'a, StencilType: TVStencil<1, 3>> Direct3Pt1DSolver<'a, StencilType> {
    pub fn new(stencil: &'a StencilType, steps: usize, threads: usize) -> Self {
        Direct3Pt1DSolver {
            solver: DirectSolver3Pt1DOpt { stencil },
            steps,
            threads,
        }
    }
}

impl<'a, StencilType: TVStencil<1, 3>> SolverInterface<1>
    for Direct3Pt1DSolver<'a, StencilType>
{
    fn apply<'b>(
        &mut self,
        input_domain: &mut SliceDomain<'b, 1>,
        output_domain: &mut SliceDomain<'b, 1>,
        mut global_time: usize,
    ) {
        debug_assert_eq!(input_domain.aabb(), output_domain.aabb());

        let n_r = input_domain.aabb().buffer_size();
        for _ in 0..self.steps - 1 {
            self.solver.apply_step(
                input_domain,
                output_domain,
                self.threads,
                global_time,
                n_r,
            );
            global_time += 1;
            std::mem::swap(input_domain, output_domain);
        }
        std::mem::swap(input_domain, output_domain);
    }

    fn print_report(&self) {
        println!("Direct3Pt1DSolver: No report");
    }

    fn to_dot_file<P: AsRef<std::path::Path>>(&self, _path: &P) {
        println!("WARNING: Direct3Pt1DSolver cannot print dot file");
    }
}
