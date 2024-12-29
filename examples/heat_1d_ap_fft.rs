use nhls::domain::*;
use nhls::image_1d_example::*;
use nhls::solver::*;

fn main() {
    let (args, output_image_path) = Args::cli_parse("heat_1d_ap_fft");

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.5);

    // Create domains
    let grid_bound = args.grid_bounds();
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);

    // Create BC
    let bc = ConstantCheck::new(1.0, grid_bound);

    // Create AP Solver
    let cutoff = 40;
    let ratio = 0.5;
    let mut solver = APSolver::new(
        &bc,
        &stencil,
        cutoff,
        ratio,
        &grid_bound,
        args.chunk_size,
    );

    // Make image
    let mut img = nhls::image::Image1D::new(grid_bound, args.lines as u32);
    img.add_line(0, input_domain.buffer());
    for t in 1..args.lines as u32 {
        solver.loop_solve(
            &mut input_domain,
            &mut output_domain,
            args.steps_per_line,
        );
        std::mem::swap(&mut input_domain, &mut output_domain);
        img.add_line(t, input_domain.buffer());
    }

    img.write(&output_image_path);
}
