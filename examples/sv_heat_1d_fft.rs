use core::f64;

use nhls::ap_solver::*;
use nhls::domain::*;
use nhls::image_1d_example::*;
use nhls::init::*;
use nhls::mirror_domain::*;
use std::time::*;

fn main() {
    let (args, output_image_path) = Args::cli_setup("sv_heat_1d_fft");

    // Grid size
    let mut grid_bound = args.grid_bounds();
    grid_bound.bounds[(0, 1)] = (args.domain_size / 2) as i32 - 1;
    println!("Grid bound: {}", grid_bound);

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.2);

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

    let direct_solver = SV1DDirectSolver::new(&stencil);
    let planner_params = PlannerParameters {
        plan_type: args.plan_type,
        cutoff: args.cutoff,
        ratio: args.ratio,
        chunk_size: args.chunk_size,
        threads: args.threads,
        steps: args.steps_per_line,
        aabb: grid_bound,
    };
    let solver = SVSolver::new(&stencil, &planner_params, direct_solver);

    if args.gen_only {
        args.finish();
        std::process::exit(0);
    }

    let mut img = None;
    if args.write_image {
        let mut i = nhls::image::Image1D::new(grid_bound, args.lines as u32);
        i.add_line(0, input_domain_1.buffer());
        img = Some(i);
    }

    // Apply direct solver
    let mut global_time = 0;
    for t in 1..args.lines {
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

        global_time += args.steps_per_line;
        std::mem::swap(&mut input_domain_1, &mut output_domain_1);
        std::mem::swap(&mut input_domain_2, &mut output_domain_2);

        if let Some(i) = img.as_mut() {
            i.add_line(t as u32, input_domain_1.buffer());
        }
    }
    if let Some(i) = img {
        i.write(&output_image_path.unwrap());
    }
    args.finish();
}
