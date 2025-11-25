use crate::domain::*;
use crate::par_stencil;
use crate::solver_interface::*;
use crate::stencil::*;
/*
pub struct MultiTVDirectSolver<
    'a,
    BC,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    BC: BCCheck<GRID_DIMENSION>,
{
    bc: &'a BC,
    stencils: [&'a TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>],
    steps: usize,
    chunk_size: usize,
    
}

impl<'a, BC, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>
    MultiTVDirectSolver<'a, BC, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    BC: BCCheck<GRID_DIMENSION>,
{
    pub fn new(
        bc: &'a BC,
        stencil: [&'a TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>],
        steps: usize,
        chunk_size: usize,
    ) -> Self {
        MultiTVDirectSolver {
            bc,
            stencil,
            steps,
            chunk_size,
        }
    }
}
*/
/*
impl<'a, BC, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>
    SolverInterface<GRID_DIMENSION>
    for GeneralDirectBoxSolver<'a, BC, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    BC: BCCheck<GRID_DIMENSION>,
{
    fn apply<'b>(
        &mut self,
        input_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        mut global_time: usize,
    ) {
        debug_assert_eq!(input_domain.aabb(), output_domain.aabb());
        for _ in 0..self.steps - 1 {
            global_time += 1;
            par_stencil::apply(
                self.bc,
                self.stencil,
                input_domain,
                output_domain,
                global_time,
                self.chunk_size,
            );
            std::mem::swap(input_domain, output_domain);
        }
        global_time += 1;
        par_stencil::apply(
            self.bc,
            self.stencil,
            input_domain,
            output_domain,
            global_time,
            self.chunk_size,
        );
    }

    fn print_report(&self) {
        println!("GeneralDirectBoxSolver: No Report");
    }

    fn to_dot_file<P: AsRef<std::path::Path>>(&self, _path: &P) {
        eprintln!("WARNING: GeneralDirectBoxSolver cannot save to dot file");
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
    global_time += 1;
    par_stencil::apply(bc, stencil, input, output, global_time, chunk_size);
}
*/
