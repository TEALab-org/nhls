use nhls::image_1d_example::*;
use nhls::time_varying::tv_periodic_solver::*;

fn main() {
    let args = Args::cli_setup("tv_heat_1d_p_fft");

    let stencil = nhls::standard_stencils::TVHeat1D::new();

    let grid_bound = args.grid_bounds();
    let mut solver = TVPeriodicSolver::new(
        &stencil,
        args.steps_per_line,
        args.plan_type,
        grid_bound,
        args.threads,
    );

    args.run_solver(&mut solver);
}
