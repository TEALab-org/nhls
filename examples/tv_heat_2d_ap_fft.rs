use core::f64;

use nhls::ap_solver::*;
use nhls::domain::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::init::*;
use std::time::*;

fn main() {
    let args = Args::cli_setup("tv_heat_2d_ap_fft");

    // Grid size
    let grid_bound = args.grid_bounds();

    let stencil = nhls::standard_stencils::TVHeat2D::new();

    // Create domains
    let mut buffer_1 = OwnedDomain::new(grid_bound);
    let mut buffer_2 = OwnedDomain::new(grid_bound);
    let mut input_domain = buffer_1.as_slice_domain();
    let mut output_domain = buffer_2.as_slice_domain();
    rand(&mut input_domain, 10, args.chunk_size);

    // This optimized direct solver implement a uniform boundary condition of 0.0
    let direct_solver = DirectSolver5Pt2DOpt::new(&stencil);

    // Create AP Solver
    let solver_params = args.solver_parameters();
    let mut solver =
        generate_tv_ap_solver(&stencil, direct_solver, &solver_params);

    if args.gen_only {
        args.finish();
        std::process::exit(0);
    }

    if args.write_images {
        image2d(&input_domain, &args.frame_name(0));
    }

    // Apply direct solver
    let mut global_time = 0;
    for t in 1..args.images {
        let now = Instant::now();
        solver.apply(&mut input_domain, &mut output_domain, global_time);
        let elapsed_time = now.elapsed();
        eprintln!("{}", elapsed_time.as_nanos() as f64 / 1000000000.0);

        global_time += args.steps_per_image;
        std::mem::swap(&mut input_domain, &mut output_domain);
        if args.write_images {
            image2d(&input_domain, &args.frame_name(t));
        }
    }

    args.finish();
}
