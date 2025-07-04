use nhls::ap_solver::*;
use nhls::direct_solver::*;
use nhls::domain::*;
use nhls::image_3d_example::*;

fn main() {
    let args = Args::cli_setup("heat_3d_ap_fft");

    let stencil =
        nhls::standard_stencils::heat_3d(1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1);

    // Create BC
    let grid_bound = args.grid_bounds();
    let bc = ConstantCheck::new(1.0, grid_bound);

    let direct_solver = DirectFrustrumSolver {
        bc: &bc,
        stencil: &stencil,
        stencil_slopes: stencil.slopes(),
        chunk_size: args.chunk_size,
    };

    // Create AP Solver
    let solver_parameters = args.solver_parameters();
    let mut solver =
        generate_ap_solver(&stencil, direct_solver, &solver_parameters);

    args.run_solver(&mut solver);
}
