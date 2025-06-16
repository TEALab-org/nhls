use nhls::domain::*;
use nhls::fft_solver::PeriodicSolver;
use nhls::image_3d_example::*;

fn main() {
    let args = Args::cli_setup("heat_3d_p_fft");

    let stencil =
        nhls::standard_stencils::heat_3d(1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1);

    // Apply periodic solver
    let grid_bound = args.grid_bounds();
    let mut buffer = OwnedDomain::new(grid_bound);
    let mut solver = PeriodicSolver::create(
        &stencil,
        buffer.buffer_mut(),
        &grid_bound,
        args.steps_per_image,
        args.plan_type,
        args.chunk_size,
        args.threads,
    );

    args.run_solver(&mut solver);
}
