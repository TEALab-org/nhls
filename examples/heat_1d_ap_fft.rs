use nhls::ap_solver::*;
use nhls::direct_solver::*;
use nhls::image_1d_example::*;

fn main() {
    let args = Args::cli_setup("heat_1d_ap_fft");

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.5);

    // This optimized direct solver implement a uniform boundary condition of 0.0
    let direct_solver = DirectSolver3Pt1DOpt::new(&stencil);

    // Create AP Solver
    let solver_params = args.solver_parameters();
    let mut solver =
        generate_ap_solver(&stencil, direct_solver, &solver_params);

    args.run_solver(&mut solver);
}
