use core::f64;

use nhls::domain::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::init::*;
use nhls::time_varying::*;
use std::time::*;

fn main() {
    let args = Args::cli_parse("tv_heat_2d_p_fft");

    // Grid size
    let grid_bound = args.grid_bounds();

    let stencil = nhls::standard_stencils::TVHeat2D::new();

    // Create domains
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);
    rand(&mut input_domain, 10, args.chunk_size);

    let mut periodic_solver = TVPeriodicSolver::new(
        &stencil,
        args.steps_per_image,
        args.plan_type,
        grid_bound,
        args.threads,
    );

    if args.gen_only {
        args.save_wisdom();
        std::process::exit(0);
    }

    if args.write_images {
        image2d(&input_domain, &args.frame_name(0));
    }

    // Apply direct solver
    let mut global_time = 0;
    for t in 1..args.images {
        let now = Instant::now();
        periodic_solver.apply(
            &mut input_domain,
            &mut output_domain,
            global_time,
        );
        let elapsed_time = now.elapsed();
        eprintln!("{}", elapsed_time.as_nanos() as f64 / 1000000000.0);

        global_time += args.steps_per_image;
        std::mem::swap(&mut input_domain, &mut output_domain);
        if args.write_images {
            image2d(&input_domain, &args.frame_name(t));
        }
    }

    args.save_wisdom();
}
