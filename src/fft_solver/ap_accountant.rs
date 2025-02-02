use crate::fft_solver::*;
use crate::util::*;
use std::ops::Range;

/// Used to calculate the memory requirement for each node in an `APPlan`.
/// See the static `node_requirements` method.
///
/// Since we require all domains to use buffers allocated in terms of
/// `MIN_ALIGMENT`, the `APAccountBuilder` calculates requirements in terms
/// of `MIN_ALIGNMENT` sized blocks.
pub struct APAccountBuilder<'a, const GRID_DIMENSION: usize> {
    plan: &'a APPlan<GRID_DIMENSION>,
}

impl<'a, const GRID_DIMENSION: usize> APAccountBuilder<'a, GRID_DIMENSION> {
    /// Generate the memory requirements for each node in MIN_ALIGMENT byte alignment
    pub fn node_requirements(plan: &'a APPlan<GRID_DIMENSION>) -> Vec<usize> {
        let mut node_requirements = vec![0; plan.len()];
        let account_builder = APAccountBuilder { plan };
        account_builder.handle_repeat_node(plan.root, &mut node_requirements);
        node_requirements
    }

    fn real_buffer_requirement(&self, aabb: &AABB<GRID_DIMENSION>) -> usize {
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
        let repeat_node = self.plan.unwrap_repeat_node(node_id);

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

    // Boundary solves each require their own Memory
    // So we sum their requirements.
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

    // For boundary solves and time cuts, we don't know the operation
    // type, only whether input / output domains are pre-allocated.
    fn handle_unknown(
        &self,
        node_id: NodeId,
        pre_allocated_io: bool,
        node_requirements: &mut [usize],
    ) -> usize {
        match self.plan.get_node(node_id) {
            PlanNode::DirectSolve(_) => self.handle_direct_node(
                node_id,
                pre_allocated_io,
                node_requirements,
            ),
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

    // Periodic Nodes memory usage:
    // - Input / output domains if not pre-allocated
    // Take the max of the following, they're mutually exclusive.
    // - Complex Buffer
    // - Memory for boundary solves
    // - Memory for timecut solve with pre-allocated input / output domains
    fn handle_periodic_node(
        &self,
        node_id: NodeId,
        pre_allocated_io: bool,
        node_requirements: &mut [usize],
    ) -> usize {
        let periodic_node = self.plan.unwrap_periodic_node(node_id);
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

    // Direct Node memory usage:
    // - Input / output domains if not pre-allocated
    pub fn handle_direct_node(
        &self,
        node_id: NodeId,
        pre_allocated_io: bool,
        node_requirements: &mut [usize],
    ) -> usize {
        let mut node_requirement = 0;
        let direct_node = self.plan.unwrap_direct_node(node_id);

        if !pre_allocated_io {
            node_requirement +=
                2 * self.real_buffer_requirement(&direct_node.input_aabb);
        }

        node_requirements[node_id] = node_requirement;
        node_requirement
    }
}
