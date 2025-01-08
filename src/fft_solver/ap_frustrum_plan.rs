use crate::domain::*;
use crate::fft_solver::*;
use crate::util::*;

pub struct FFTSolveNode<const GRID_DIMENSION: usize> {
    pub boundary_solve: NodeId,
    pub next: Option<NodeId>,
    pub input_aabb: AABB<GRID_DIMENSION>,
    pub output_aabb: AABB<GRID_DIMENSION>,
    pub convolution_id: OpId,
    pub steps: usize,
}

pub struct DirectSolveNode<const GRID_DIMENSION: usize> {
    pub input_aabb: AABB<GRID_DIMENSION>,
    pub output_aabb: AABB<GRID_DIMENSION>,
    pub steps: usize,
}

pub enum Node<const GRID_DIMENSION: usize> {
    FFTSolve(FFTSolveNode<GRID_DIMENSION>),
    DirectSolve(DirectSolveNode<GRID_DIMENSION>),
}

pub struct APFrustrumPlan<const GRID_DIMENSION: usize> {
    nodes: Vec<Node<GRID_DIMENSION>>,
    //stencil_slopes: Bounds<GRID_DIMENSION>,
    //frustrum_slopes: Bounds<GRID_DIMENSION>,
    root: NodeId,
}

impl<const GRID_DIMENSION: usize> APFrustrumPlan<GRID_DIMENSION> {
    fn get_node(&self, node: NodeId) -> &Node<GRID_DIMENSION> {
        &self.nodes[node]
    }
    /*
        pub fn execute<DomainType: DomainView<GRID_DIMENSION>>(
            &self,
            solve_input_domain: &DomainType,
            solve_output_domain: &mut DomainType,
            buffer_domain: &mut DomainType,
            complex_buffer: &mut [c64],
            op_store: &ConvolutionStore,
            chunk_size: usize,
        ) {
            debug_assert_eq!(solve_input_domain.aabb(), solve_output_domain.aabb());
            debug_assert_eq!(solve_input_domain.aabb(), buffer_domain.aabb());
            debug_assert_eq!(solve_input_domain.aabb().complex_buffer_size(), complex_buffer.len());
            // We need a stack
            let mut stack = vec![self.root];
            while let Some(root) = stack.pop() {
                match self.get_node(root) {
                    Node::FFTSolve(fft_solve) => {
                        // Apply convolution from input to buffer,
                        let convolution_op = op_store.get(fft_solve.convolution_id);
                        convolution_op.apply(input_domain, output_domain, complex_buffer, chunk_size);
                        output_domain.par_set_subdomain(
                        // copy from buffer into output

                    }
                    Node::DirectSolve(direct_solve) => {

                    }
                }
            }
        }
    */
}
