use crate::ap_solver::find_periodic_solve::*;
use crate::ap_solver::frustrum::*;
use crate::ap_solver::index_types::*;
use crate::ap_solver::periodic_ops::*;
use crate::ap_solver::plan::*;
use crate::ap_solver::solver_parameters::*;
use crate::util::*;

/// Creating a plan results in both a plan and convolution store.
/// Someday we may separate the creation, if for example we
/// we add support for saving Plans to file.
pub struct PlannerResult<
    const GRID_DIMENSION: usize,
    PeriodicOpsType: PeriodicOps<GRID_DIMENSION>,
> {
    pub plan: Plan<GRID_DIMENSION>,
    pub periodic_ops: PeriodicOpsType,
    pub remainder_periodic_ops: PeriodicOpsType,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
}

/// Used to create an `Plan`. See `create_ap_plan`
pub struct Planner<
    'a,
    const GRID_DIMENSION: usize,
    PeriodicOpsType: PeriodicOps<GRID_DIMENSION>,
    OpsBuilderType: PeriodicOpsBuilder<GRID_DIMENSION, PeriodicOpsType>,
> {
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
    pub nodes: Vec<PlanNode<GRID_DIMENSION>>,
    pub params: &'a SolverParameters<GRID_DIMENSION>,
    pub ops_builder: OpsBuilderType,
    pub ops_type_marker: std::marker::PhantomData<PeriodicOpsType>,
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        PeriodicOpsType: PeriodicOps<GRID_DIMENSION>,
        OpsBuilderType: PeriodicOpsBuilder<GRID_DIMENSION, PeriodicOpsType>,
    > Planner<'a, GRID_DIMENSION, PeriodicOpsType, OpsBuilderType>
{
    /// Pushes a node into the plan store and returns the id.
    pub fn add_node(&mut self, node: PlanNode<GRID_DIMENSION>) -> NodeId {
        let result = self.nodes.len();
        self.nodes.push(node);
        result
    }

    /// Create a direct solve node for a given frustrum.
    fn generate_direct_node(
        &mut self,
        frustrum: Frustrum<GRID_DIMENSION>,
        tasks: usize,
    ) -> PlanNode<GRID_DIMENSION> {
        let input_aabb = frustrum.input_aabb(&self.stencil_slopes);
        let direct_node = DirectSolveNode {
            input_aabb,
            output_aabb: frustrum.output_aabb,
            sloped_sides: frustrum.sloped_sides(),
            steps: frustrum.steps,
            threads: tasks,
        };
        PlanNode::DirectSolve(direct_node)
    }

    fn generate_periodic_node(
        &mut self,
        mut frustrum: Frustrum<GRID_DIMENSION>,
        periodic_solve: PeriodicSolve<GRID_DIMENSION>,
        rel_time_0: usize,
        tasks: usize,
    ) -> PlanNode<GRID_DIMENSION> {
        // Create the convolution operation and get the id
        let input_aabb = frustrum.input_aabb(&self.stencil_slopes);
        let rel_time_post = rel_time_0 + periodic_solve.steps;
        let op_descriptor = PeriodicOpDescriptor {
            step_min: rel_time_0,
            step_max: rel_time_post,
            steps: periodic_solve.steps,
            exclusive_bounds: input_aabb.exclusive_bounds(),
            threads: tasks,
        };
        let convolution_id = self.ops_builder.get_op_id(op_descriptor);

        // Do we need a time cut.
        // If so that will trim that and create a plan for it
        let mut time_cut = None;
        let maybe_next_frustrum =
            frustrum.time_cut(periodic_solve.steps, &self.stencil_slopes);
        if let Some(next_frustrum) = maybe_next_frustrum {
            let next_node =
                self.generate_frustrum(next_frustrum, rel_time_post, tasks);
            time_cut = Some(self.add_node(next_node));
        }
        debug_assert!(frustrum
            .output_aabb
            .contains_aabb(&periodic_solve.output_aabb));

        // Generate nodes for the boundary solves
        let boundary_frustrums = frustrum.decompose(&self.stencil_slopes);
        let mut sub_nodes = Vec::with_capacity(2 * GRID_DIMENSION);
        let sub_tasks = self
            .params
            .task_min
            .max((tasks as f64 / boundary_frustrums.len() as f64).ceil()
                as usize);

        for bf in boundary_frustrums {
            sub_nodes.push(self.generate_frustrum(bf, rel_time_0, sub_tasks));
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
            threads: tasks,
        })
    }

    /// Create node, possibly by adding other nodes,
    /// to handle a frustrum.
    /// This is where we try to find a periodic solve, and may create
    /// a boundary decomposition.
    fn generate_frustrum(
        &mut self,
        frustrum: Frustrum<GRID_DIMENSION>,
        rel_time_0: usize,
        tasks: usize,
    ) -> PlanNode<GRID_DIMENSION> {
        let solve_params = PeriodicSolveParams {
            stencil_slopes: self.stencil_slopes,
            cutoff: self.params.cutoff,
            ratio: self.params.ratio,
            max_steps: Some(frustrum.steps),
        };
        let input_aabb = frustrum.input_aabb(&self.stencil_slopes);
        debug_assert!(self.params.aabb.contains_aabb(&input_aabb));

        // Can we do a periodic solve or do we direct solve?
        let maybe_periodic_solve =
            find_periodic_solve(&input_aabb, &solve_params);

        if let Some(periodic_solve) = maybe_periodic_solve {
            self.generate_periodic_node(
                frustrum,
                periodic_solve,
                rel_time_0,
                tasks,
            )
        } else {
            self.generate_direct_node(frustrum, tasks)
        }
    }

    /// The root AABB requires special treatment.
    /// This function creates a plan for the larges periodic solve
    /// it can find within the box and max_steps.
    ///
    /// Note also that the boundary solve decomposition
    /// is based on `AABB` and not `Frustrum`.
    pub fn generate_central(
        &mut self,
        max_steps: usize,
        threads: usize,
    ) -> (NodeId, usize) {
        let rel_time_0 = 0;
        let solve_params = PeriodicSolveParams {
            stencil_slopes: self.stencil_slopes,
            cutoff: self.params.cutoff,
            ratio: self.params.ratio,
            max_steps: Some(max_steps),
        };

        let periodic_solve =
            find_periodic_solve(&self.params.aabb, &solve_params).unwrap();

        let op_descriptor = PeriodicOpDescriptor {
            step_min: 0,
            step_max: periodic_solve.steps,
            steps: periodic_solve.steps,
            exclusive_bounds: self.params.aabb.exclusive_bounds(),
            threads,
        };
        let convolution_id = self.ops_builder.get_op_id(op_descriptor);

        let decomposition =
            self.params.aabb.decomposition(&periodic_solve.output_aabb);
        let mut sub_nodes = Vec::with_capacity(2 * GRID_DIMENSION);
        let sub_tasks = self.params.task_min.max(
            (self.params.task_mult * threads as f64 / GRID_DIMENSION as f64)
                .ceil() as usize,
        );

        for d in 0..GRID_DIMENSION {
            for side in [Side::Min, Side::Max] {
                sub_nodes.push(self.generate_frustrum(
                    Frustrum::new(
                        decomposition[d][side.outer_index()],
                        d,
                        side,
                        periodic_solve.steps,
                    ),
                    rel_time_0,
                    sub_tasks,
                ));
            }
        }

        // add the nodes, find the range
        let first_node = self.nodes.len();
        let last_node = first_node + sub_nodes.len();
        self.nodes.extend(&mut sub_nodes.drain(..));

        let periodic_solve_node = PeriodicSolveNode {
            input_aabb: self.params.aabb,
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
}
