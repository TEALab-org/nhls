/// We track memory in terms of 128bit alignments
/// per MIN_ALIGNMENT in fftw3 build

use crate::fft_solver::*;
use crate::util::*;
use std::ops::Range;

pub struct APAccountBuilder<'a, const GRID_DIMENSION: usize> {
    pub plan: &'a APPlan<GRID_DIMENSION>,
    pub node_requirements: Vec<usize>,
}

pub struct APAccountant<'a, const GRID_DIMENSION: usize> {
    pub plan: &'a APPlan<GRID_DIMENSION>,
}


impl<'a, const GRID_DIMENSION: usize> APAccountant<'a, GRID_DIMENSION> {
    pub fn scratch_size(plan: &'a APPlan<GRID_DIMENSION>) -> usize {
        let ac = APAccountant {
            plan,
        };

        ac.handle_repeat_node(plan.root)
    }

    pub fn get_input_output_size(&self, aabb: &AABB<GRID_DIMENSION>) -> usize {
        // We deal in complex sizes to keep things aligned.
        // however, we need to buffers of real numbers of the same size,
        // so just one buffer size for complex
        aabb.buffer_size()
    }

    pub fn handle_repeat_node(&self, node: NodeId) -> usize {
        let repeat_node = if let PlanNode::Repeat(repeat_node) = self.plan.get_node(node) {
        repeat_node
        } else {
            panic!("ERROR: Not a repeat node");
        };

        let mut result = self.handle_central_periodic_node(repeat_node.node);
        if let Some(next) = repeat_node.next {
            result = result.max(self.handle_central_periodic_node(next));
        }
        println!("Repeat: {}", result);

        result 
    }

    pub fn handle_remainder(&self, node_range: Range<NodeId>) -> usize {
        let mut sum = 0;
        for node_id in node_range {
            sum += self.handle_unknown(node_id); 
        }
        sum
    }

    pub fn handle_unknown(&self, node_id: NodeId) -> usize {
        match self.plan.get_node(node_id) {
                PlanNode::DirectSolve(_) => self.handle_direct_node(node_id),
                PlanNode::PeriodicSolve(_) => self.handle_periodic_node(node_id),
                PlanNode::Repeat(_) => {
                    panic!("ERROR: Not expecting repeat node");
                },
            }
    }

    // The central periodic solves already have input / output domains
    // allocated.
    // We just need the complex buffer
    pub fn handle_central_periodic_node(&self, node: NodeId) -> usize {
        let periodic_node = if let PlanNode::PeriodicSolve(periodic_node) = self.plan.get_node(node) {
            periodic_node
        } else {
            panic!("ERROR: Not a periodic node: {}", node);
        };

        let remainder = self.handle_remainder(periodic_node.remainder.clone());
        let complex = periodic_node.input_aabb.complex_buffer_size();
        let mut result = remainder.max(complex);
        if let Some(time_cut) = periodic_node.time_cut {
            let cut = self.handle_unknown(time_cut);
            result = result.max(cut);
            println!("handle central with cut: remainder: {}, complex: {}, cut: {}, result: {}", remainder, complex, cut, result);
        } else {
            println!("handle central: remainder: {}, complex: {}, result: {}", remainder, complex, result);
        }

        result
    }

    // Allocate input / output domains, complex buffer, then handle remainder
    // next operation doesn't need input output allocated, can use the current one
    // maybe use handle_central_periodic for that?
    pub fn handle_periodic_node(&self, node: NodeId) -> usize {
        let periodic_node = if let PlanNode::PeriodicSolve(periodic_node) = self.plan.get_node(node) {
            periodic_node
        } else {
            panic!("ERROR: Not a periodic node");
        };

        let mut result = self.get_input_output_size(&periodic_node.input_aabb);
        let remainder = self.handle_remainder(periodic_node.remainder.clone());
        let complex = periodic_node.input_aabb.complex_buffer_size();
        result += remainder.max(complex);
        if let Some(time_cut) = periodic_node.time_cut {
            let cut = self.handle_unknown(time_cut);
            result = result.max(cut);
            println!("handle periodic with cut: remainder: {}, complex: {}, cut: {}, result: {}", remainder, complex, cut, result);
        } else {
            println!("handle periodic: remainder: {}, complex: {}, result: {}", remainder, complex, result);
        }

        result

    }

    // We just need input / output node allocated.
    pub fn handle_direct_node(&self, node: NodeId) -> usize {
        let direct_node = if let PlanNode::DirectSolve(direct_node) = self.plan.get_node(node) {
            direct_node
        } else {
            panic!("ERROR: Not a periodic node");
        };

        let result = self.get_input_output_size(&direct_node.input_aabb);
        println!("handle direct, result: {}", result);
        result
    }
}

