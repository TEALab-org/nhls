use nhls::domain::*;
use nhls::fft_solver::PeriodicSolver;
use nhls::image_1d_example::*;
use nhls::init;

fn main() {
    let args = Args::cli_setup("heat_1d_p_fft");

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.5);

    // Create domains
    let grid_bound = args.grid_bounds();
    let mut buffer_1 = OwnedDomain::new(grid_bound);
    let mut buffer_2 = OwnedDomain::new(grid_bound);
    let mut input_domain = buffer_1.as_slice_domain();
    let mut output_domain = buffer_2.as_slice_domain();

    let mut solver = PeriodicSolver::create(
        &stencil,
        output_domain.buffer_mut(),
        &grid_bound,
        args.steps_per_line,
        args.plan_type,
        args.chunk_size,
        args.threads,
    );

    args.run_solver_with_domains(
        &mut input_domain,
        &mut output_domain,
        &mut solver,
    );
}
