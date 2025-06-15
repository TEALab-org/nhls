use nhls::domain::*;
use nhls::fft_solver::PeriodicSolver;
use nhls::image_1d_example::*;

fn main() {
    let args = Args::cli_setup("heat_1d_p_fft");

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.5);

    // Create solver
    let grid_bound = args.grid_bounds();
    let mut buffer = OwnedDomain::new(grid_bound);
    let mut solver = PeriodicSolver::create(
        &stencil,
        buffer.buffer_mut(),
        &grid_bound,
        args.steps_per_line,
        args.plan_type,
        args.chunk_size,
        args.threads,
    );

    args.run_solver(&mut solver);
}
