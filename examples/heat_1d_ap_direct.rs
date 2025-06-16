use nhls::domain::*;
use nhls::image_1d_example::*;
use nhls::solver::*;

fn main() {
    let args = Args::cli_setup("heat_1d_ap_direct");

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.5);

    // Create BC
    let grid_bound = args.grid_bounds();
    let bc = ConstantCheck::new(1.0, grid_bound);

    let mut solver = GeneralDirectBoxSolver::new(
        &bc,
        &stencil,
        args.steps_per_line,
        args.chunk_size,
    );

    args.run_solver(&mut solver);
}
