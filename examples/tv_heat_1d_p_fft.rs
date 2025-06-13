use core::f64;

use nhls::domain::*;
use nhls::image_1d_example::*;
use nhls::init::*;
use nhls::time_varying::tv_periodic_solver::*;
use std::time::*;

fn main() {
    let (args, output_image_path) = Args::cli_setup("tv_heat_1d_p_fft");

    // Grid size
    let grid_bound = args.grid_bounds();

    let stencil = nhls::standard_stencils::TVHeat1D::new();

    // Create domains
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);
    rand(&mut input_domain, 10, args.chunk_size);

    let mut periodic_solver = TVPeriodicSolver::new(
        &stencil,
        args.steps_per_line,
        args.plan_type,
        grid_bound,
        args.threads,
    );

    if args.gen_only {
        args.finish();
        std::process::exit(0);
    }

    let mut img = None;
    if args.write_image {
        let mut i = nhls::image::Image1D::new(grid_bound, args.lines as u32);
        i.add_line(0, input_domain.buffer());
        img = Some(i);
    }

    // Apply direct solver
    let mut global_time = 0;
    for t in 1..args.lines {
        let now = Instant::now();

        periodic_solver.apply(
            &mut input_domain,
            &mut output_domain,
            global_time,
        );

        let elapsed_time = now.elapsed();

        eprintln!("{}", elapsed_time.as_nanos() as f64 / 1000000000.0);
        global_time += args.steps_per_line;
        std::mem::swap(&mut input_domain, &mut output_domain);
        if let Some(i) = img.as_mut() {
            i.add_line(t as u32, input_domain.buffer());
        }
    }

    if let Some(i) = img {
        i.write(&output_image_path);
    }

    args.finish();
}
