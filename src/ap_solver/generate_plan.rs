use crate::ap_solver::periodic_ops::*;
use crate::ap_solver::plan::*;
use crate::ap_solver::planner::*;
use crate::stencil::TVStencil;

/// Create the root repeat node.
pub fn generate_plan<
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    PeriodicOpsType: PeriodicOps<GRID_DIMENSION>,
    OpsBuilderType: PeriodicOpsBuilder<GRID_DIMENSION, PeriodicOpsType>,
    CreateBuilderFn: Fn() -> OpsBuilderType,
>(
    stencil: &StencilType,
    create_builder: CreateBuilderFn,
    params: &PlannerParameters<GRID_DIMENSION>,
) -> PlannerResult<GRID_DIMENSION, PeriodicOpsType> {
    let stencil_slopes = stencil.slopes();
    let nodes = Vec::new();
    let mut planner = Planner {
        stencil_slopes,
        params,
        nodes,
        ops_builder: create_builder(),
        ops_type_marker: std::marker::PhantomData,
    };
    // generate central once,
    let (central_solve_node, central_solve_steps) =
        planner.generate_central(params.steps, params.threads);

    let n = params.steps / central_solve_steps;
    let remainder = params.steps % central_solve_steps;
    let mut next = None;
    let mut t_builder = create_builder();
    std::mem::swap(&mut planner.ops_builder, &mut t_builder);
    let periodic_ops = t_builder.finish();

    if remainder != 0 {
        let (remainder_solve_node, remainder_solve_steps) =
            planner.generate_central(remainder, params.threads);
        next = Some(remainder_solve_node);
        debug_assert_eq!(remainder_solve_steps, remainder);
    }

    t_builder = create_builder();
    std::mem::swap(&mut planner.ops_builder, &mut t_builder);
    let remainder_periodic_ops = t_builder.finish();

    let repeat_node = RepeatNode {
        n,
        node: central_solve_node,
        next,
    };

    let root = planner.add_node(PlanNode::Repeat(repeat_node));

    let plan = Plan {
        nodes: planner.nodes,
        root,
    };

    PlannerResult {
        plan,
        periodic_ops,
        remainder_periodic_ops,
        stencil_slopes,
    }
}
