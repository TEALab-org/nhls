use nhls::ap_solver::*;
use nhls::image_2d_example::*;

fn main() {
    let args = Args::cli_setup("heat_2d_ap_fft");

    let stencil = nhls::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);

    // This optimized direct solver implement a uniform boundary condition of 0.0
    let direct_solver = DirectSolver5Pt2DOpt::new(&stencil);

    // Create AP Solver
    let solver_params = args.solver_parameters();
    let mut solver =
        generate_ap_solver_2d(&stencil, direct_solver, &solver_params);

    args.run_solver(&mut solver);
}
