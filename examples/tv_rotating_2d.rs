use core::f64;

use nhls::ap_solver::*;
use nhls::domain::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::init::*;
use std::time::*;

fn main() {
    let args = Args::cli_parse("tv_rotating_2d");

    // Grid size
    let grid_bound = args.grid_bounds();

    let freq = (2.0 * std::f64::consts::PI) / 1500.0;
    let stencil =
        nhls::standard_stencils::RotatingAdvectionStencil::new(freq, 0.2);

    // Create domains
    let mut buffer_1 = OwnedDomain::new(grid_bound);
    let mut buffer_2 = OwnedDomain::new(grid_bound);
    let mut input_domain = buffer_1.as_slice_domain();
    let mut output_domain = buffer_2.as_slice_domain();
    normal_ic_2d(&mut input_domain, args.chunk_size);

    let direct_solver = DirectSolver5Pt2DOpt::new(&stencil);
    // Create AP Solver
    let planner_params = PlannerParameters {
        plan_type: args.plan_type,
        cutoff: args.cutoff,
        ratio: args.ratio,
        chunk_size: args.chunk_size,
        threads: args.threads,
        steps: args.steps_per_image,
        aabb: grid_bound,
    };
    let mut solver =
        generate_tv_ap_solver(&stencil, direct_solver, &planner_params);

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
        solver.apply(&mut input_domain, &mut output_domain, global_time);
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
