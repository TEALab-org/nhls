use crate::fft_solver::ap_frustrum::*;
use crate::fft_solver::find_periodic_solve::*;
use crate::fft_solver::PlanType;

use crate::stencil::*;

use crate::ap_solver::index_types::*;
use crate::ap_solver::periodic_ops::*;
use crate::ap_solver::plan::*;

use crate::util::*;

/// Planner recurses by finding periodic solves.
/// These solves are configured with these parameters.
pub struct PlannerParameters {
    pub plan_type: PlanType,
    pub cutoff: i32,
    pub ratio: f64,
    pub chunk_size: usize,
    pub solve_threads: usize,
}

/// Creating a plan results in both a plan and convolution store.
/// Someday we may separate the creation, if for example we
/// we add support for saving APPlans to file.
pub struct PlannerResult<
    const GRID_DIMENSION: usize,
    PeriodicOpsType: PeriodicOps<GRID_DIMENSION>,
> {
    pub plan: Plan<GRID_DIMENSION>,
    pub periodic_ops: PeriodicOpsType,
    pub remainder_periodic_ops: PeriodicOpsType,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
}

/// Used to create an `APPlan`. See `create_ap_plan`
pub struct Planner<
    const GRID_DIMENSION: usize,
    PeriodicOpsType: PeriodicOps<GRID_DIMENSION>,
    OpsBuilderType: PeriodicOpsBuilder<GRID_DIMENSION, PeriodicOpsType>,
> {
    stencil_slopes: Bounds<GRID_DIMENSION>,
    aabb: AABB<GRID_DIMENSION>,
    steps: usize,
    ops_builder: OpsBuilderType,
    nodes: Vec<PlanNode<GRID_DIMENSION>>,

    cutoff: i32,
    ratio: f64,

    ops_type_marker: std::marker::PhantomData<PeriodicOpsType>,
}

impl<
        const GRID_DIMENSION: usize,
        PeriodicOpsType: PeriodicOps<GRID_DIMENSION>,
        OpsBuilderType: PeriodicOpsBuilder<GRID_DIMENSION, PeriodicOpsType>,
    > Planner<GRID_DIMENSION, PeriodicOpsType, OpsBuilderType>
{
    pub fn new<
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    >(
        stencil: &StencilType,
        aabb: AABB<GRID_DIMENSION>,
        steps: usize,
        cutoff: i32,
        ratio: f64,
    ) -> Self {
        let stencil_slopes = stencil.slopes();
        let ops_builder = OpsBuilderType::blank();
        let nodes = Vec::new();
        Planner {
            stencil_slopes,
            aabb,
            steps,
            cutoff,
            ratio,
            ops_builder,
            nodes,
            ops_type_marker: std::marker::PhantomData,
        }
    }

    /// Pushes a node into the plan store and returns the id.
    fn add_node(&mut self, node: PlanNode<GRID_DIMENSION>) -> NodeId {
        let result = self.nodes.len();
        self.nodes.push(node);
        result
    }

    /// Create a direct solve node for a given frustrum.
    fn generate_direct_node(
        &mut self,
        frustrum: APFrustrum<GRID_DIMENSION>,
        threads: usize,
    ) -> PlanNode<GRID_DIMENSION> {
        let input_aabb = frustrum.input_aabb(&self.stencil_slopes);
        let direct_node = DirectSolveNode {
            input_aabb,
            output_aabb: frustrum.output_aabb,
            sloped_sides: frustrum.sloped_sides(),
            steps: frustrum.steps,
            threads,
        };
        PlanNode::DirectSolve(direct_node)
    }

    fn generate_periodic_node(
        &mut self,
        mut frustrum: APFrustrum<GRID_DIMENSION>,
        periodic_solve: PeriodicSolve<GRID_DIMENSION>,
        rel_time_0: usize,
        threads: usize,
    ) -> PlanNode<GRID_DIMENSION> {
        // Create the convolution operation and get the id
        let input_aabb = frustrum.input_aabb(&self.stencil_slopes);
        let rel_time_post = rel_time_0 + periodic_solve.steps;
        let op_descriptor = PeriodicOpDescriptor {
            step_min: rel_time_0,
            step_max: rel_time_post,
            steps: periodic_solve.steps,
            exclusive_bounds: input_aabb.exclusive_bounds(),
            threads,
        };
        let convolution_id = self.ops_builder.get_op_id(op_descriptor);

        // Do we need a time cut.
        // If so that will trim that and create a plan for it
        let mut time_cut = None;
        let maybe_next_frustrum =
            frustrum.time_cut(periodic_solve.steps, &self.stencil_slopes);
        if let Some(next_frustrum) = maybe_next_frustrum {
            let next_node =
                self.generate_frustrum(next_frustrum, rel_time_post, threads);
            time_cut = Some(self.add_node(next_node));
        }
        debug_assert!(frustrum
            .output_aabb
            .contains_aabb(&periodic_solve.output_aabb));

        // Generate nodes for the boundary solves
        let boundary_frustrums = frustrum.decompose(&self.stencil_slopes);
        let mut sub_nodes = Vec::with_capacity(2 * GRID_DIMENSION);
        let sub_threads = 1
            .max((threads as f64 / boundary_frustrums.len() as f64).ceil()
                as usize);

        for bf in boundary_frustrums {
            sub_nodes.push(self.generate_frustrum(bf, rel_time_0, sub_threads));
        }

        // Ensure boundary solve nodes are contiguous in the plan
        let first_node = self.nodes.len();
        let last_node = first_node + sub_nodes.len();
        self.nodes.extend(&mut sub_nodes.drain(..));

        PlanNode::PeriodicSolve(PeriodicSolveNode {
            input_aabb,
            output_aabb: frustrum.output_aabb,
            convolution_id,
            steps: periodic_solve.steps,
            boundary_nodes: first_node..last_node,
            time_cut,
            threads,
        })
    }

    /// Create node, possibly by adding other nodes,
    /// to handle a frustrum.
    /// This is where we try to find a periodic solve, and may create
    /// a boundary decomposition.
    fn generate_frustrum(
        &mut self,
        frustrum: APFrustrum<GRID_DIMENSION>,
        rel_time_0: usize,
        threads: usize,
    ) -> PlanNode<GRID_DIMENSION> {
        let solve_params = PeriodicSolveParams {
            stencil_slopes: self.stencil_slopes,
            cutoff: self.cutoff,
            ratio: self.ratio,
            max_steps: Some(frustrum.steps),
        };
        let input_aabb = frustrum.input_aabb(&self.stencil_slopes);
        debug_assert!(self.aabb.contains_aabb(&input_aabb));

        // Can we do a periodic solve or do we direct solve?
        let maybe_periodic_solve =
            find_periodic_solve(&input_aabb, &solve_params);
        if maybe_periodic_solve.is_none() {
            self.generate_direct_node(frustrum, threads)
        } else {
            let periodic_solve = maybe_periodic_solve.unwrap();
            self.generate_periodic_node(
                frustrum,
                periodic_solve,
                rel_time_0,
                threads,
            )
        }
    }

    /// The root AABB requires special treatment.
    /// This function creates a plan for the larges periodic solve
    /// it can find within the box and max_steps.
    ///
    /// Note also that the boundary solve decomposition
    /// is based on `AABB` and not `APFrustrum`.
    fn generate_central(
        &mut self,
        max_steps: usize,
        threads: usize,
    ) -> (NodeId, usize) {
        let rel_time_0 = 0;
        let solve_params = PeriodicSolveParams {
            stencil_slopes: self.stencil_slopes,
            cutoff: self.cutoff,
            ratio: self.ratio,
            max_steps: Some(max_steps),
        };

        let periodic_solve =
            find_periodic_solve(&self.aabb, &solve_params).unwrap();

        let op_descriptor = PeriodicOpDescriptor {
            step_min: 0,
            step_max: periodic_solve.steps,
            steps: periodic_solve.steps,
            exclusive_bounds: self.aabb.exclusive_bounds(),
            threads,
        };
        let convolution_id = self.ops_builder.get_op_id(op_descriptor);

        let decomposition =
            self.aabb.decomposition(&periodic_solve.output_aabb);
        let mut sub_nodes = Vec::with_capacity(2 * GRID_DIMENSION);
        let sub_threads =
            1.max(
                (threads as f64 / (GRID_DIMENSION * 2) as f64).ceil() as usize
            );

        for d in 0..GRID_DIMENSION {
            for side in [Side::Min, Side::Max] {
                sub_nodes.push(self.generate_frustrum(
                    APFrustrum::new(
                        decomposition[d][side.outer_index()],
                        d,
                        side,
                        periodic_solve.steps,
                    ),
                    rel_time_0,
                    sub_threads,
                ));
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
            boundary_nodes: first_node..last_node,
            time_cut: None,
            threads,
        };

        let root_node =
            self.add_node(PlanNode::PeriodicSolve(periodic_solve_node));

        (root_node, periodic_solve.steps)
    }

    /// Create the root repeat node.
    fn generate(
        &mut self,
        threads: usize,
    ) -> (NodeId, PeriodicOpsType, PeriodicOpsType) {
        // generate central once,
        let (central_solve_node, central_solve_steps) =
            self.generate_central(self.steps, threads);

        let n = self.steps / central_solve_steps;
        let remainder = self.steps % central_solve_steps;
        let mut next = None;
        let mut remainder_periodic_ops = PeriodicOpsType::blank();

        let mut t_builder = OpsBuilderType::blank();
        std::mem::swap(&mut self.ops_builder, &mut t_builder);
        let periodic_ops = t_builder.finish();

        if remainder != 0 {
            let (remainder_solve_node, remainder_solve_steps) =
                self.generate_central(remainder, threads);
            next = Some(remainder_solve_node);
            let mut t_builder = OpsBuilderType::blank();
            std::mem::swap(&mut self.ops_builder, &mut t_builder);
            remainder_periodic_ops = t_builder.finish();
            debug_assert_eq!(remainder_solve_steps, remainder);
        }

        let repeat_node = RepeatNode {
            n,
            node: central_solve_node,
            next,
        };

        let root_node = self.add_node(PlanNode::Repeat(repeat_node));
        (root_node, periodic_ops, remainder_periodic_ops)
    }

    /// Package up the results
    pub fn finish(
        mut self,
        threads: usize,
    ) -> PlannerResult<GRID_DIMENSION, PeriodicOpsType> {
        let (root, periodic_ops, remainder_periodic_ops) =
            self.generate(threads);
        let stencil_slopes = self.stencil_slopes;
        let plan = Plan {
            nodes: self.nodes,
            root,
        };

        PlannerResult {
            plan,
            periodic_ops,
            remainder_periodic_ops,
            stencil_slopes,
        }
    }
}
