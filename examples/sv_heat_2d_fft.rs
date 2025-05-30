use core::f64;

use nhls::domain::*;
use nhls::fft_solver::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::init::*;
use nhls::mirror_domain::*;
use std::time::*;

fn main() {
    let args = Args::cli_parse("sv_heat_2d_fft");

    // Grid size
    let mut grid_bound = args.grid_bounds();
    grid_bound.bounds[(0, 1)] = (args.domain_size / 2) as i32 - 1;
    println!("Grid bound: {}", grid_bound);

    let stencil = nhls::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);

    // Create domains
    let mut buffer_11 = OwnedDomain::new(grid_bound);
    let mut buffer_12 = OwnedDomain::new(grid_bound);
    let mut input_domain_1 = buffer_11.as_slice_domain();
    let mut output_domain_1 = buffer_12.as_slice_domain();
    rand(&mut input_domain_1, 10, args.chunk_size);

    let mut buffer_21 = OwnedDomain::new(grid_bound);
    let mut buffer_22 = OwnedDomain::new(grid_bound);
    let mut input_domain_2 = buffer_21.as_slice_domain();
    let mut output_domain_2 = buffer_22.as_slice_domain();
    rand(&mut input_domain_2, 10, args.chunk_size);

    let direct_solver = SV2DDirectSolver::new(&stencil);
    let planner_params = PlannerParameters {
        plan_type: args.plan_type,
        cutoff: args.cutoff,
        ratio: args.ratio,
        chunk_size: args.chunk_size,
        solve_threads: args.threads,
    };
    let solver = SVSolver::new(
        &stencil,
        grid_bound,
        args.steps_per_image,
        &planner_params,
        direct_solver,
    );

    if args.gen_only {
        args.save_wisdom();
        std::process::exit(0);
    }

    if args.write_images {
        image2d(&input_domain_1, &args.frame_name(0));
    }

    // Apply direct solver
    let mut global_time = 0;
    for t in 1..args.images {
        let now = Instant::now();
        solver.apply(
            &mut input_domain_1,
            &mut output_domain_1,
            &mut input_domain_2,
            &mut output_domain_2,
            global_time,
        );
        let elapsed_time = now.elapsed();
        eprintln!("{}", elapsed_time.as_nanos() as f64 / 1000000000.0);

        global_time += args.steps_per_image;
        std::mem::swap(&mut input_domain_1, &mut output_domain_1);
        std::mem::swap(&mut input_domain_2, &mut output_domain_2);

        if args.write_images {
            image2d(&input_domain_1, &args.frame_name(t));
        }
    }

    args.save_wisdom();
}
