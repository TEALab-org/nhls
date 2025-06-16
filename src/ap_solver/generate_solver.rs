use crate::ap_solver::ap_periodic_ops_builder::*;
use crate::ap_solver::direct_solver::*;
use crate::ap_solver::generate_plan::*;
use crate::ap_solver::scratch_builder::ComplexBufferType;
use crate::ap_solver::solver::*;
use crate::ap_solver::solver_parameters::*;
use crate::ap_solver::tv_periodic_ops_collector::*;
use crate::domain::*;
use crate::stencil::*;
use crate::SolverInterface;

pub fn generate_ap_solver_1d<
    const NEIGHBORHOOD_SIZE: usize,
    DirectSolverType: DirectSolver<1>,
>(
    stencil: &Stencil<1, NEIGHBORHOOD_SIZE>,
    direct_solver: DirectSolverType,
    params: &SolverParameters<1>,
) -> impl SolverInterface<1> {
    let create_ops_builder = || ApPeriodicOpsBuilder::new(stencil, params);
    let planner_result = generate_plan(stencil, create_ops_builder, params);
    let complex_buffer_type = ComplexBufferType::DomainOnly;
    let subset_ops = SubsetOps1d {};
    Solver::new(
        direct_solver,
        subset_ops,
        params,
        planner_result,
        complex_buffer_type,
    )
}

pub fn generate_ap_solver_2d<
    const NEIGHBORHOOD_SIZE: usize,
    DirectSolverType: DirectSolver<2>,
>(
    stencil: &Stencil<2, NEIGHBORHOOD_SIZE>,
    direct_solver: DirectSolverType,
    params: &SolverParameters<2>,
) -> impl SolverInterface<2> {
    let create_ops_builder = || ApPeriodicOpsBuilder::new(stencil, params);
    let planner_result = generate_plan(stencil, create_ops_builder, params);
    let complex_buffer_type = ComplexBufferType::DomainOnly;
    let subset_ops = SubsetOps2d {};
    Solver::new(
        direct_solver,
        subset_ops,
        params,
        planner_result,
        complex_buffer_type,
    )
}

pub fn generate_ap_solver_3d<
    const NEIGHBORHOOD_SIZE: usize,
    DirectSolverType: DirectSolver<3>,
>(
    stencil: &Stencil<3, NEIGHBORHOOD_SIZE>,
    direct_solver: DirectSolverType,
    params: &SolverParameters<3>,
) -> impl SolverInterface<3> {
    let create_ops_builder = || ApPeriodicOpsBuilder::new(stencil, params);
    let planner_result = generate_plan(stencil, create_ops_builder, params);
    let complex_buffer_type = ComplexBufferType::DomainOnly;
    let subset_ops = SubsetOps3d {};
    Solver::new(
        direct_solver,
        subset_ops,
        params,
        planner_result,
        complex_buffer_type,
    )
}

pub fn generate_tv_ap_solver_1d<
    'a, 
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<1, NEIGHBORHOOD_SIZE>,
    DirectSolverType: DirectSolver<1> + 'a,
>(
    stencil: &'a StencilType,
    direct_solver: DirectSolverType,
    params: &'a SolverParameters<1>,
) -> impl SolverInterface<1> + 'a {
    let create_ops_builder = || TvPeriodicOpsCollector::new(stencil, params);
    let planner_result = generate_plan(stencil, create_ops_builder, params);
    let complex_buffer_type = ComplexBufferType::DomainAndOp;
    let subset_ops = SubsetOps1d {};
    Solver::new(
        direct_solver,
        subset_ops,
        params,
        planner_result,
        complex_buffer_type,
    )
}

pub fn generate_tv_ap_solver_2d<
    'a, 
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<2, NEIGHBORHOOD_SIZE>,
    DirectSolverType: DirectSolver<2> + 'a,
>(
    stencil: &'a StencilType,
    direct_solver: DirectSolverType,
    params: &'a SolverParameters<2>,
) -> impl SolverInterface<2> + 'a {
    let create_ops_builder = || TvPeriodicOpsCollector::new(stencil, params);
    let planner_result = generate_plan(stencil, create_ops_builder, params);
    let complex_buffer_type = ComplexBufferType::DomainAndOp;
    let subset_ops = SubsetOps2d {};
    Solver::new(
        direct_solver,
        subset_ops,
        params,
        planner_result,
        complex_buffer_type,
    )
}
