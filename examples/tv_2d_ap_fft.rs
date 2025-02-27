use nhls::domain::*;
use nhls::fft_solver::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::init::*;
use nhls::solver::*;
use nhls::time_varying::*;
//use nhls::util::*;

fn main() {
    let args = Args::cli_parse("tv_2d_ap_fft");

    // Grid size
    let grid_bound = args.grid_bounds();

    let freq = (2.0 * std::f64::consts::PI) / 1500.0;
    let stencil =
        nhls::standard_stencils::RotatingAdvectionStencil::new(freq, 0.1);

    // Create domains
    // Create domains
    let mut buffer_1 = OwnedDomain::new(grid_bound);
    let mut buffer_2 = OwnedDomain::new(grid_bound);
    let mut input_domain = buffer_1.as_slice_domain();
    let mut output_domain = buffer_2.as_slice_domain();
    normal_ic_2d(&mut input_domain, args.chunk_size);

    if args.write_images {
        image2d(&input_domain, &args.frame_name(0));
    }

    let solver = AP2DDirectSolver::new(&stencil);
    let planner_params = PlannerParameters {
        plan_type: args.plan_type,
        cutoff: args.cutoff,
        ratio: args.ratio,
        chunk_size: args.chunk_size,
    };
    let mut tv_ap_solver = TVAPSolver::new(
        &stencil,
        grid_bound,
        args.steps_per_image,
        args.threads,
        &planner_params,
        solver,
    );
    if args.write_dot {
        tv_ap_solver.to_dot_file(&args.dot_path());
    }

    if args.gen_only {
        args.save_wisdom();
        std::process::exit(0);
    }

    // Apply fft solver
    let mut global_time = 0;
    for t in 1..args.images {
        tv_ap_solver.apply(&mut input_domain, &mut output_domain, global_time);
        global_time += args.steps_per_image;
        std::mem::swap(&mut input_domain, &mut output_domain);
        if args.write_images {
            image2d(&input_domain, &args.frame_name(t));
        }
    }
}
