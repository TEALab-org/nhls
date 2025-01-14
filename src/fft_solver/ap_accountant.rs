/// We track memory in terms of 128bit alignments
/// per MIN_ALIGNMENT in fftw3 build
use crate::fft_solver::*;
use crate::util::*;
use std::ops::Range;

pub const MIN_ALIGNMENT: usize = 128;

pub struct APAccountBuilder<'a, const GRID_DIMENSION: usize> {
    pub plan: &'a APPlan<GRID_DIMENSION>,
}

impl<'a, const GRID_DIMENSION: usize> APAccountBuilder<'a, GRID_DIMENSION> {
    /// Generate the memory requirements for each node in 128bit alignment
    pub fn node_requirements(plan: &'a APPlan<GRID_DIMENSION>) -> Vec<usize> {
        let mut node_requirements = vec![0; plan.len()];
        let account_builder = APAccountBuilder { plan };
        account_builder.handle_repeat_node(plan.root, &mut node_requirements);
        node_requirements
    }

    fn real_buffer_requirement(&self, aabb: &AABB<GRID_DIMENSION>) -> usize {
        // TODO: this is where we should do the 128bit offset thing
        // We are allocating two, but the more I think about it,
        // we should just always allocate 128 aligned things
        let min_bytes = aabb.buffer_size() * std::mem::size_of::<f64>();
        min_bytes.div_ceil(MIN_ALIGNMENT)
    }

    fn complex_buffer_requirement(&self, aabb: &AABB<GRID_DIMENSION>) -> usize {
        let min_bytes = aabb.complex_buffer_size() * std::mem::size_of::<c64>();
        min_bytes.div_ceil(MIN_ALIGNMENT)
    }

    fn handle_repeat_node(
        &self,
        node_id: NodeId,
        node_requirements: &mut [usize],
    ) -> usize {
        let repeat_node = if let PlanNode::Repeat(repeat_node) =
            self.plan.get_node(node_id)
        {
            repeat_node
        } else {
            panic!("ERROR: Not a repeat node");
        };

        let mut node_requirement = self.handle_periodic_node(
            repeat_node.node,
            true,
            node_requirements,
        );
        if let Some(next) = repeat_node.next {
            node_requirement = node_requirement.max(self.handle_periodic_node(
                next,
                true,
                node_requirements,
            ));
        }
        node_requirements[node_id] = node_requirement;
        node_requirement
    }

    fn handle_boundary_operations(
        &self,
        node_range: Range<NodeId>,
        node_requirements: &mut [usize],
    ) -> usize {
        let mut sum = 0;
        for node_id in node_range {
            sum += self.handle_unknown(node_id, false, node_requirements);
        }
        sum
    }

    fn handle_unknown(
        &self,
        node_id: NodeId,
        pre_allocated_io: bool,
        node_requirements: &mut [usize],
    ) -> usize {
        match self.plan.get_node(node_id) {
            PlanNode::DirectSolve(_) => {
                self.handle_direct_node(node_id, node_requirements)
            }
            PlanNode::PeriodicSolve(_) => self.handle_periodic_node(
                node_id,
                pre_allocated_io,
                node_requirements,
            ),
            PlanNode::Repeat(_) => {
                panic!("ERROR: Not expecting repeat node");
            }
        }
    }
    // Allocate input / output domains, complex buffer, then handle boundary
    // next operation doesn't need input output allocated, can use the current one
    // maybe use handle_central_periodic for that?
    fn handle_periodic_node(
        &self,
        node_id: NodeId,
        pre_allocated_io: bool,
        node_requirements: &mut [usize],
    ) -> usize {
        let periodic_node = if let PlanNode::PeriodicSolve(periodic_node) =
            self.plan.get_node(node_id)
        {
            periodic_node
        } else {
            panic!("ERROR: Not a periodic node");
        };

        let remainder = self.handle_boundary_operations(
            periodic_node.boundary_nodes.clone(),
            node_requirements,
        );
        let complex =
            self.complex_buffer_requirement(&periodic_node.input_aabb);
        let mut node_requirement = remainder.max(complex);
        if let Some(time_cut) = periodic_node.time_cut {
            // Time cuts can re-use io buffers
            let pre_allocated_io = true;
            let cut_requirement = self.handle_unknown(
                time_cut,
                pre_allocated_io,
                node_requirements,
            );
            node_requirement = node_requirement.max(cut_requirement);
        }

        if !pre_allocated_io {
            node_requirement +=
                2 * self.real_buffer_requirement(&periodic_node.input_aabb);
        }

        node_requirements[node_id] = node_requirement;
        node_requirement
    }

    // We just need input / output node allocated.
    pub fn handle_direct_node(
        &self,
        node_id: NodeId,
        node_requirements: &mut [usize],
    ) -> usize {
        let direct_node = if let PlanNode::DirectSolve(direct_node) =
            self.plan.get_node(node_id)
        {
            direct_node
        } else {
            panic!("ERROR: Not a periodic node");
        };

        let node_requirement =
            2 * self.real_buffer_requirement(&direct_node.input_aabb);
        node_requirements[node_id] = node_requirement;
        node_requirement
    }
}
