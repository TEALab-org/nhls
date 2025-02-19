use core::f64;

use nhls::domain::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::init::*;
use nhls::time_varying::*;
use nhls::util::*;

fn main() {
    let args = Args::cli_parse("tv_2d_p_fft");

    // Grid size
    let grid_bound = args.grid_bounds();

    let stencil = nhls::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);

    // Create domains
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);
    normal_ic_2d(&mut input_domain, args.chunk_size);

    if args.write_images {
        image2d(&input_domain, &args.frame_name(0));
    }

    let mut p = TVTreePlanner::new(&stencil, grid_bound);
    p.build_range(0, args.steps_per_image, 0);
    if args.write_dot {
        p.to_dot_file(&args.tree_dot_path());
        println!("max layer: {}", p.max_layer);
    }

    let mut tv_planner = TVPlanner::new(
        &stencil,
        grid_bound,
        args.steps_per_image,
        args.cutoff,
        args.ratio,
    );
    tv_planner.generate();
    let tv_result = tv_planner.finish();

    if args.write_dot {
        p.to_dot_file(&args.tree_dot_path());
        tv_result.plan.to_dot_file(&args.dot_path());
        println!("max layer: {}", p.max_layer);
    }

    if args.gen_only {
        std::process::exit(0);
    }
}
