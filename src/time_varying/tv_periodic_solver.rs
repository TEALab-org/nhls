use crate::domain::*;
use crate::fft_solver::*;
use crate::par_slice;
use crate::time_varying::*;
use crate::util::*;
use fftw::plan::*;

pub struct FFTPlanPair {
    pub forward_plan: fftw::plan::Plan<f64, c64, fftw::plan::Plan64>,
    pub backward_plan: fftw::plan::Plan<c64, f64, fftw::plan::Plan64>,
}

pub struct TVPeriodicSolver<
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> {
    tree: TVTree<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>,
    steps: usize,
    threads: usize,
    pub forward_plan: fftw::plan::Plan<f64, c64, fftw::plan::Plan64>,
    pub backward_plan: fftw::plan::Plan<c64, f64, fftw::plan::Plan64>,
    s_complex_buffer: AlignedVec<c64>,
    i_complex_buffer: AlignedVec<c64>,
    chunk_size: usize,
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    > TVPeriodicSolver<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
{
    pub fn new(
        stencil: &'a StencilType,
        steps: usize,
        plan_type: PlanType,
        aabb: AABB<GRID_DIMENSION>,
        threads: usize,
        chunk_size: usize,
    ) -> Self {
        let size = aabb.exclusive_bounds();
        let plan_size = size.try_cast::<usize>().unwrap();
        let forward_plan = fftw::plan::R2CPlan64::aligned(
            plan_size.as_slice(),
            plan_type.to_fftw3_flag(),
        )
        .unwrap();
        let backward_plan = fftw::plan::C2RPlan64::aligned(
            plan_size.as_slice(),
            plan_type.to_fftw3_flag(),
        )
        .unwrap();

        let s_complex_buffer = AlignedVec::new(aabb.complex_buffer_size());
        let i_complex_buffer = AlignedVec::new(aabb.complex_buffer_size());

        TVPeriodicSolver {
            tree: TVTree::new(stencil, aabb),
            steps,
            threads,
            forward_plan,
            backward_plan,
            s_complex_buffer,
            i_complex_buffer,
            chunk_size,
        }
    }

    pub fn apply<DomainType: DomainView<GRID_DIMENSION>>(
        &mut self,
        input: &mut DomainType,
        output: &mut DomainType,
        global_time: usize,
    ) {
        debug_assert_eq!(*input.aabb(), self.tree.aabb);
        debug_assert_eq!(*output.aabb(), self.tree.aabb);
        let s = self.tree.build_range(global_time, global_time + self.steps);
        let aabb = self.tree.aabb;
        let mut s_domain = OwnedDomain::new(aabb);
        for (offset, weight) in s.to_offset_weights() {
            let rn_i: Coord<GRID_DIMENSION> = aabb.min() + offset * -1;
            let periodic_coord = aabb.periodic_coord(&rn_i);
            s_domain.set_coord(&periodic_coord, weight);
        }

        // forward pass s_domain
        self.forward_plan
            .r2c(s_domain.buffer_mut(), &mut self.s_complex_buffer)
            .unwrap();

        // forward pass input
        self.forward_plan
            .r2c(input.buffer_mut(), &mut self.i_complex_buffer)
            .unwrap();

        // mul
        par_slice::multiply_by(
            &mut self.s_complex_buffer,
            &self.i_complex_buffer,
            self.chunk_size,
        );

        // backward pass output
        self.backward_plan
            .c2r(&mut self.s_complex_buffer, output.buffer_mut())
            .unwrap();

        let n_r = output.aabb().buffer_size();
        par_slice::div(output.buffer_mut(), n_r as f64, self.chunk_size);
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::mem_fmt::human_readable_bytes;

    use super::*;

    fn test_tree_size<const GRID_DIMENSION: usize>(
        stencil_slopes: Bounds<GRID_DIMENSION>,
        steps: usize,
    ) {
        let mut current_nodes = steps;
        let mut current_slopes = stencil_slopes;
        let mut layer = 1;
        while current_nodes != 1 {
            let new_layer_size = current_nodes / 2;
            let extra_node = current_nodes % 2 == 1;
            current_slopes += current_slopes;
            let aabb = slopes_to_circ_aabb(&current_slopes);
            // 2 input domains, one complex domain
            let node_size = 2 * aabb.buffer_size() * size_of::<f64>()
                + 2 * aabb.complex_buffer_size() * size_of::<c64>();
            let size = new_layer_size * node_size;
            println!("layer: {}, current_nodes: {}, new_layer_size: {}, extra_node: {}, current_slopes: {:?}, size: {}, {}", layer, current_nodes, new_layer_size, extra_node, current_slopes, size, human_readable_bytes(size));
            current_nodes = new_layer_size + extra_node as usize;
            layer += 1;
        }
    }

    #[test]
    fn test_stencil_size() {
        {
            println!("steps 10, 2d");
            let slopes = matrix![1, 1; 1, 1];
            let steps = 8000;
            test_tree_size(slopes, steps);
        }
    }
}
