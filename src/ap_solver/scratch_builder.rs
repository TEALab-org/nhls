use crate::ap_solver::account_builder::*;
use crate::ap_solver::index_types::*;
use crate::ap_solver::plan::*;
use crate::ap_solver::scratch::*;
use crate::ap_solver::MIN_ALIGNMENT;
use crate::util::*;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ComplexBufferType {
    // Periodic Ops only
    DomainOnly,

    // TV operations need buffer space to build the convolution
    DomainAndOp,
}

/// `APScratchBuilder` calculates offsets and sizes for the scratch memory
/// each node will use in terms of bytes.
/// See `APScratchBuilder::build` method.
/// These offsets will respect `MIN_ALIGMENT`.
/// Note that these buffers are provided at runtime by `APScratch`,
/// the scratch builder just creates a descriptor for each node in
/// the plan.
pub struct ScratchBuilder<'a, const GRID_DIMENSION: usize> {
    plan: &'a Plan<GRID_DIMENSION>,
    node_block_requirements: Vec<usize>,
    complex_buffer_type: ComplexBufferType,
}

impl<'a, const GRID_DIMENSION: usize> ScratchBuilder<'a, GRID_DIMENSION> {
    /// Static method to create an `APScratch` instance
    /// and a scratch descriptor for each node
    pub fn build(
        plan: &'a Plan<GRID_DIMENSION>,
        complex_buffer_type: ComplexBufferType,
    ) -> (Vec<ScratchDescriptor>, Scratch) {
        let node_block_requirements =
            AccountBuilder::node_requirements(plan, complex_buffer_type);
        let mut scratch_descriptors =
            vec![ScratchDescriptor::default(); plan.len()];

        let builder = ScratchBuilder {
            plan,
            node_block_requirements,
            complex_buffer_type,
        };
        builder.handle_repeat(plan.root, 0, &mut scratch_descriptors);
        let scratch_space = Scratch::new(
            builder.blocks_to_bytes(builder.node_block_requirements[plan.root]),
        );
        (scratch_descriptors, scratch_space)
    }

    pub fn build_double(
        plan: &'a Plan<GRID_DIMENSION>,
        complex_buffer_type: ComplexBufferType,
    ) -> (Vec<ScratchDescriptor>, Scratch, Scratch) {
        let node_block_requirements =
            AccountBuilder::node_requirements(plan, complex_buffer_type);
        let mut scratch_descriptors =
            vec![ScratchDescriptor::default(); plan.len()];

        let builder = ScratchBuilder {
            plan,
            node_block_requirements,
            complex_buffer_type,
        };
        builder.handle_repeat(plan.root, 0, &mut scratch_descriptors);
        let scratch_space_1 = Scratch::new(
            builder.blocks_to_bytes(builder.node_block_requirements[plan.root]),
        );
        let scratch_space_2 = Scratch::new(
            builder.blocks_to_bytes(builder.node_block_requirements[plan.root]),
        );
        (scratch_descriptors, scratch_space_1, scratch_space_2)
    }

    fn blocks_to_bytes(&self, blocks: usize) -> usize {
        blocks * MIN_ALIGNMENT
    }

    fn real_buffer_bytes(&self, aabb: &AABB<GRID_DIMENSION>) -> usize {
        let min_bytes = aabb.buffer_size() * std::mem::size_of::<f64>();
        min_bytes.div_ceil(MIN_ALIGNMENT) * MIN_ALIGNMENT
    }

    fn complex_buffer_bytes(&self, aabb: &AABB<GRID_DIMENSION>) -> usize {
        let min_bytes = aabb.complex_buffer_size() * std::mem::size_of::<c64>();
        let byte_req = min_bytes.div_ceil(MIN_ALIGNMENT) * MIN_ALIGNMENT;
        match self.complex_buffer_type {
            ComplexBufferType::DomainOnly => byte_req,
            ComplexBufferType::DomainAndOp => 2 * byte_req,
        }
    }

    fn handle_repeat(
        &self,
        node_id: NodeId,
        offset: usize,
        scratch_descriptors: &mut [ScratchDescriptor],
    ) {
        let repeat_node = self.plan.unwrap_repeat_node(node_id);

        self.handle_periodic(
            repeat_node.node,
            offset,
            true,
            scratch_descriptors,
        );
        if let Some(next) = repeat_node.next {
            self.handle_unknown(next, offset, true, scratch_descriptors);
        }
    }

    fn handle_unknown(
        &self,
        node_id: NodeId,
        offset: usize,
        pre_allocated_io: bool,
        scratch_descriptors: &mut [ScratchDescriptor],
    ) {
        match self.plan.get_node(node_id) {
            PlanNode::DirectSolve(_) => self.handle_direct(
                node_id,
                offset,
                pre_allocated_io,
                scratch_descriptors,
            ),
            PlanNode::PeriodicSolve(_) => self.handle_periodic(
                node_id,
                offset,
                pre_allocated_io,
                scratch_descriptors,
            ),
            PlanNode::Repeat(_) => {
                panic!("ERROR: Not expecting repeat node");
            }
            PlanNode::Range(_) => {
                panic!("ERROR: Not expecting range node");
            }
        }
    }

    fn handle_direct(
        &self,
        node_id: NodeId,
        offset: usize,
        pre_allocated_io: bool,
        scratch_descriptors: &mut [ScratchDescriptor],
    ) {
        let direct_solve = self.plan.unwrap_direct_node(node_id);
        let scratch_descriptor = &mut scratch_descriptors[node_id];

        // Input / Output scratch?
        if !pre_allocated_io {
            let buffer_len = self.real_buffer_bytes(&direct_solve.input_aabb);
            scratch_descriptor.input_offset = offset;
            scratch_descriptor.output_offset = offset + buffer_len;
            scratch_descriptor.real_buffer_size = buffer_len;
        }
    }

    fn handle_periodic(
        &self,
        node_id: NodeId,
        mut offset: usize,
        pre_allocated_io: bool,
        scratch_descriptors: &mut [ScratchDescriptor],
    ) {
        let periodic_solve = self.plan.unwrap_periodic_node(node_id);
        let scratch_descriptor = &mut scratch_descriptors[node_id];

        // Input / Output scratch?
        if !pre_allocated_io {
            let buffer_len = self.real_buffer_bytes(&periodic_solve.input_aabb);
            scratch_descriptor.input_offset = offset;
            scratch_descriptor.output_offset = offset + buffer_len;
            scratch_descriptor.real_buffer_size = buffer_len;
            offset += 2 * buffer_len;
        }

        // Complex buffer scratch?
        let complex_buffer_len =
            self.complex_buffer_bytes(&periodic_solve.input_aabb);
        scratch_descriptor.complex_offset = offset;
        scratch_descriptor.complex_buffer_size = complex_buffer_len;

        // Boundary solves scratch
        // Each boundary solve needs to allocate io buffers
        let mut boundary_offset = offset;
        for boundary_node in periodic_solve.boundary_nodes.clone() {
            self.handle_unknown(
                boundary_node,
                boundary_offset,
                false,
                scratch_descriptors,
            );
            boundary_offset += self
                .blocks_to_bytes(self.node_block_requirements[boundary_node]);
        }

        // Time Cut
        if let Some(time_cut) = periodic_solve.time_cut {
            // Time cuts can re-use io buffers
            let pre_allocated_io = true;
            self.handle_unknown(
                time_cut,
                offset,
                pre_allocated_io,
                scratch_descriptors,
            );
        }
    }
}
