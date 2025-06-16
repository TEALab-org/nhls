use nhls::ap_solver::*;
use nhls::image_2d_example::*;

fn main() {
    let args = Args::cli_setup("tv_rotating_2d");

    let freq = (2.0 * std::f64::consts::PI) / 1500.0;
    let stencil =
        nhls::standard_stencils::RotatingAdvectionStencil::new(freq, 0.2);

    let direct_solver = DirectSolver5Pt2DOpt::new(&stencil);

    // Create AP Solver
    let solver_params = args.solver_parameters();
    let mut solver =
        generate_tv_ap_solver(&stencil, direct_solver, &solver_params);

    args.run_solver(&mut solver);
}
