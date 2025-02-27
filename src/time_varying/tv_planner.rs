use crate::fft_solver::*;
use crate::time_varying::*;
use crate::util::*;
/// Creating a plan results in both a plan and convolution store.
/// Someday we may separate the creation, if for example we
/// we add support for saving APPlans to file.
pub struct TVPlannerResult<const GRID_DIMENSION: usize> {
    pub plan: APPlan<GRID_DIMENSION>,
    pub tree_query_collector: TVTreeQueryCollector<GRID_DIMENSION>,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
}

/// Given a stencil and AABB domain
/// create an `PlannerResult`.
/// We assume all faces of the AABB are boundary conditions.
pub fn create_tv_ap_plan<
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
>(
    stencil: &StencilType,
    aabb: AABB<GRID_DIMENSION>,
    steps: usize,
    params: &PlannerParameters,
) -> TVPlannerResult<GRID_DIMENSION> {
    let planner = TVPlanner::new(
        stencil,
        aabb,
        steps,
        //params.plan_type,
        params.cutoff,
        params.ratio,
        //params.chunk_size,
    );
    planner.finish()
}

/// Used to create an `APPlan`. See `create_ap_plan`
struct TVPlanner<const GRID_DIMENSION: usize> {
    stencil_slopes: Bounds<GRID_DIMENSION>,
    aabb: AABB<GRID_DIMENSION>,
    steps: usize,
    cutoff: i32,
    ratio: f64,
    tree_query_collector: TVTreeQueryCollector<GRID_DIMENSION>,
    nodes: Vec<PlanNode<GRID_DIMENSION>>,
}

impl<const GRID_DIMENSION: usize> TVPlanner<GRID_DIMENSION> {
    fn new<
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    >(
        stencil: &StencilType,
        aabb: AABB<GRID_DIMENSION>,
        steps: usize,
        //plan_type: PlanType,
        cutoff: i32,
        ratio: f64,
        //chunk_size: usize,
    ) -> Self {
        let stencil_slopes = stencil.slopes();
        let tree_query_collector = TVTreeQueryCollector::new();
        let nodes = Vec::new();
        TVPlanner {
            stencil_slopes,
            aabb,
            steps,
            cutoff,
            ratio,
            tree_query_collector,
            nodes,
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
    ) -> PlanNode<GRID_DIMENSION> {
        let input_aabb = frustrum.input_aabb(&self.stencil_slopes);
        let direct_node = DirectSolveNode {
            input_aabb,
            output_aabb: frustrum.output_aabb,
            sloped_sides: frustrum.sloped_sides(),
            steps: frustrum.steps,
        };
        PlanNode::DirectSolve(direct_node)
    }

    fn generate_periodic_node(
        &mut self,
        mut frustrum: APFrustrum<GRID_DIMENSION>,
        periodic_solve: PeriodicSolve<GRID_DIMENSION>,
        rel_time_0: usize,
    ) -> PlanNode<GRID_DIMENSION> {
        // Create the convolution operation and get the id
        let input_aabb = frustrum.input_aabb(&self.stencil_slopes);
        let rel_time_post = rel_time_0 + periodic_solve.steps;
        let op_descriptor = TVOpDescriptor {
            step_min: rel_time_0,
            step_max: rel_time_post,
            exclusive_bounds: input_aabb.exclusive_bounds(),
        };
        let convolution_id = self.tree_query_collector.get_op_id(op_descriptor);

        // Do we need a time cut.
        // If so that will trim that and create a plan for it
        let mut time_cut = None;
        let maybe_next_frustrum =
            frustrum.time_cut(periodic_solve.steps, &self.stencil_slopes);
        if let Some(next_frustrum) = maybe_next_frustrum {
            let next_node =
                self.generate_frustrum(next_frustrum, rel_time_post);
            time_cut = Some(self.add_node(next_node));
        }
        debug_assert!(frustrum
            .output_aabb
            .contains_aabb(&periodic_solve.output_aabb));

        // Generate nodes for the boundary solves
        let boundary_frustrums = frustrum.decompose(&self.stencil_slopes);
        let mut sub_nodes = Vec::with_capacity(2 * GRID_DIMENSION);
        for bf in boundary_frustrums {
            sub_nodes.push(self.generate_frustrum(bf, rel_time_0));
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
            self.generate_direct_node(frustrum)
        } else {
            let periodic_solve = maybe_periodic_solve.unwrap();
            self.generate_periodic_node(frustrum, periodic_solve, rel_time_0)
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
        rel_time_0: usize,
    ) -> (PlanNode<GRID_DIMENSION>, usize) {
        let solve_params = PeriodicSolveParams {
            stencil_slopes: self.stencil_slopes,
            cutoff: self.cutoff,
            ratio: self.ratio,
            max_steps: Some(max_steps),
        };

        let periodic_solve =
            find_periodic_solve(&self.aabb, &solve_params).unwrap();

        let op_descriptor = TVOpDescriptor {
            step_min: rel_time_0,
            step_max: rel_time_0 + periodic_solve.steps,
            exclusive_bounds: self.aabb.exclusive_bounds(),
        };
        let convolution_id = self.tree_query_collector.get_op_id(op_descriptor);

        let decomposition =
            self.aabb.decomposition(&periodic_solve.output_aabb);
        let mut sub_nodes = Vec::with_capacity(2 * GRID_DIMENSION);
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
        };

        (
            PlanNode::PeriodicSolve(periodic_solve_node),
            periodic_solve.steps,
        )
    }

    /// Create the root repeat node.
    fn generate(&mut self) -> NodeId {
        let mut root_solves = Vec::new();
        let mut rel_time_0 = 0;
        let mut remaining_time = self.steps;
        while remaining_time > 0 {
            let (central_solve_node, central_solve_steps) =
                self.generate_central(remaining_time, rel_time_0);

            root_solves.push(central_solve_node);
            rel_time_0 += central_solve_steps;
            remaining_time -= central_solve_steps;
        }

        // add the nodes, find the range
        let first_node = self.nodes.len();
        let last_node = first_node + root_solves.len();
        self.nodes.extend(&mut root_solves.drain(..));

        let range_node = RangeNode {
            range: first_node..last_node,
        };

        self.add_node(PlanNode::Range(range_node))
    }

    /// Package up the results
    fn finish(mut self) -> TVPlannerResult<GRID_DIMENSION> {
        let root = self.generate();
        let stencil_slopes = self.stencil_slopes;
        let plan = APPlan {
            nodes: self.nodes,
            root,
        };

        TVPlannerResult {
            plan,
            tree_query_collector: self.tree_query_collector,
            stencil_slopes,
        }
    }
}
