use crate::fft_solver::*;
use crate::stencil::*;
use crate::util::*;

// generic over stencil
pub struct APPlanner<
    'a,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
{
    stencil_slopes: Bounds<GRID_DIMENSION>,
    aabb: AABB<GRID_DIMENSION>,
    steps: usize,
    cutoff: i32,
    ratio: f64,
    convolution_gen:
        ConvolutionGenerator<'a, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    pub nodes: Vec<PlanNode<GRID_DIMENSION>>,
}

impl<
        'a,
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    > APPlanner<'a, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
{
    pub fn new(
        stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        aabb: AABB<GRID_DIMENSION>,
        steps: usize,
        plan_type: PlanType,
        cutoff: i32,
        ratio: f64,
        chunk_size: usize,
    ) -> Self {
        let stencil_slopes = stencil.slopes();
        let convolution_gen =
            ConvolutionGenerator::new(&aabb, stencil, plan_type, chunk_size);
        let nodes = Vec::new();
        APPlanner {
            stencil_slopes,
            aabb,
            steps,
            cutoff,
            ratio,
            convolution_gen,
            nodes,
        }
    }

    pub fn add_node(&mut self, node: PlanNode<GRID_DIMENSION>) -> NodeId {
        let result = self.nodes.len();
        self.nodes.push(node);
        result
    }
    pub fn generate_frustrum(
        &mut self,
        frustrum: APFrustrum<GRID_DIMENSION>,
    ) -> PlanNode<GRID_DIMENSION> {
        frustrum.generate_direct_node(&self.stencil_slopes)
    }

    pub fn generate_central(&mut self, max_steps: usize) -> (NodeId, usize) {
        let solve_params = PeriodicSolveParams {
            stencil_slopes: self.stencil_slopes,
            cutoff: self.cutoff,
            ratio: self.ratio,
            max_steps: Some(max_steps),
        };

        let periodic_solve =
            find_periodic_solve(&self.aabb, &solve_params).unwrap();

        let convolution_id = self
            .convolution_gen
            .get_op(&self.aabb, periodic_solve.steps);

        println!("aabb: {:?}", self.aabb);
        println!("output: {:?}", periodic_solve.output_aabb);

        let decomposition =
            self.aabb.decomposition(&periodic_solve.output_aabb);
        let mut sub_nodes = Vec::with_capacity(2 * GRID_DIMENSION);
        for d in 0..GRID_DIMENSION {
            for side in [Side::Min, Side::Max] {
                sub_nodes.push(self.generate_frustrum(APFrustrum::new(
                    decomposition[d][side.outer_index()],
                    d,
                    side,
                    periodic_solve.steps,
                )));
            }
        }

        // add the nodes, find the range
        let first_node = self.nodes.len();
        let last_node = first_node + sub_nodes.len();
        self.nodes.extend(&mut sub_nodes.drain(..));

        let periodic_solve_node = PeriodicSolveNode {
            input_aabb: self.aabb,
            output_aabb: periodic_solve.output_aabb,
            convolution_id,
            steps: periodic_solve.steps,
            remainder: first_node..last_node,
            next: None,
        };

        let root_node =
            self.add_node(PlanNode::PeriodicSolve(periodic_solve_node));
        (root_node, periodic_solve.steps)
    }

    pub fn generate(&mut self) -> NodeId {
        // generate central once,
        let (central_solve_node, central_solve_steps) =
            self.generate_central(self.steps);

        let n = self.steps / central_solve_steps;
        let remainder = self.steps % central_solve_steps;
        println!("n: {}, remainder: {}", n, remainder);
        let mut next = None;
        if remainder != 0 {
            let (remainder_solve_node, remainder_solve_steps) =
                self.generate_central(remainder);
            next = Some(remainder_solve_node);
            debug_assert_eq!(remainder_solve_steps, remainder);
        }

        let repeat_node = RepeatNode {
            n,
            node: central_solve_node,
            next,
        };

        self.add_node(PlanNode::Repeat(repeat_node))
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::standard_stencils::*;

    #[test]
    fn test_gen() {
        let stencil = heat_1d(1.0, 1.0, 0.5);
        let aabb = AABB::new(matrix![0, 100]);
        let cutoff = 20;
        let ratio = 0.5;
        let steps = 104;
        let plan_type = PlanType::Estimate;
        let chunk_size = 10;

        let mut planner = APPlanner::new(
            &stencil, aabb, steps, plan_type, cutoff, ratio, chunk_size,
        );
        planner.generate();

        for (i, n) in planner.nodes.iter().enumerate() {
            println!("i: {}, n: {:?}", i, n);
        }
    }

    #[test]
    fn decomp_test_1() {
        let aabb = AABB::new(matrix![0, 100]);
        let center = AABB::new(matrix![25, 75]);
        let d = aabb.decomposition(&center);
        println!("{:?}", d);
    }
}
