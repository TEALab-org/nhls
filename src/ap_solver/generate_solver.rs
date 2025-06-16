use crate::ap_solver::ap_periodic_ops_builder::*;
use crate::ap_solver::direct_solver::*;
use crate::ap_solver::generate_plan::*;
use crate::ap_solver::scratch_builder::ComplexBufferType;
use crate::ap_solver::solver::*;
use crate::ap_solver::solver_parameters::*;
use crate::ap_solver::tv_periodic_ops_collector::*;
use crate::stencil::*;
use crate::SolverInterface;

pub fn generate_ap_solver<
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    DirectSolverType: DirectSolver<GRID_DIMENSION>,
>(
    stencil: &Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    direct_solver: DirectSolverType,
    params: &SolverParameters<GRID_DIMENSION>,
) -> impl SolverInterface<GRID_DIMENSION> {
    let create_ops_builder = || ApPeriodicOpsBuilder::new(stencil, params);
    let planner_result = generate_plan(stencil, create_ops_builder, params);
    let complex_buffer_type = ComplexBufferType::DomainOnly;
    let solver =
        Solver::new(direct_solver, params, planner_result, complex_buffer_type);
    solver
}

pub fn generate_tv_ap_solver<
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    DirectSolverType: DirectSolver<GRID_DIMENSION> + 'a,
>(
    stencil: &'a StencilType,
    direct_solver: DirectSolverType,
    params: &'a SolverParameters<GRID_DIMENSION>,
) -> impl SolverInterface<GRID_DIMENSION> + 'a {
    let create_ops_builder = || TvPeriodicOpsCollector::new(stencil, params);
    let planner_result = generate_plan(stencil, create_ops_builder, params);
    let complex_buffer_type = ComplexBufferType::DomainAndOp;
    let solver =
        Solver::new(direct_solver, params, planner_result, complex_buffer_type);
    solver
}
