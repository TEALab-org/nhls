use nhls::direct_solver::*;
use nhls::domain::*;
use nhls::image_2d_example::*;

fn main() {
    let args = Args::cli_setup("heat_2d_ap_direct");

    let stencil = nhls::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);

    // Create BC
    let grid_bound = args.grid_bounds();
    let bc = ConstantCheck::new(1.0, grid_bound);

    let mut solver = GeneralDirectBoxSolver::new(
        &bc,
        &stencil,
        args.steps_per_image,
        args.chunk_size,
    );

    args.run_solver(&mut solver);
}
