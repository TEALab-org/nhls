use nhls::direct_solver::*;
use nhls::image_1d_example::*;

fn main() {
    let args = Args::cli_setup("heat_1d_p_direct");

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.5);

    let mut solver = GeneralDirectPeriodicBoxSolver::new(
        &stencil,
        args.steps_per_line,
        args.chunk_size,
    );

    args.run_solver(&mut solver);
}
