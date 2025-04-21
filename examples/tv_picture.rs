use core::f64;

use nhls::fft_solver::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::image_init::*;
use nhls::init::*;
use nhls::csv::*;
use nhls::solver::*;
use nhls::time_varying::*;
use std::time::*;

pub fn s_normal_ic_2d<DomainType: DomainView<2>>(
    domain: &mut DomainType,
    x_offset: f64,
    y_offset: f64,
    chunk_size: usize,
) {
    let exclusive_bounds = domain.aabb().exclusive_bounds();
    let width_f = exclusive_bounds[0] as f64;
    let height_f = exclusive_bounds[1] as f64;
    let sigma_sq: f64 = (width_f / 16.0) * (width_f / 16.0);
    let ic_gen = |coord: Coord<2>| {
        let x = (coord[0] as f64) - (width_f / 2.0) + x_offset;
        let y = (coord[1] as f64) - (height_f / 2.0) + y_offset;
        let r = (x * x + y * y).sqrt();
        let exp = -r * r / (2.0 * sigma_sq);
        exp.exp()
    };
    domain.par_set_values(ic_gen, chunk_size);
}

fn main() {
    let args = Args::cli_parse("tv_picture");

    let mut image_prob = ImageProb::load(
        &"/Users/russell/projects/stencil_research/poster_images/logo_init.png",
        args.chunk_size,
    );

    // Grid size
    let grid_bound = image_prob.aabb;

    let y_offset = 300.0;
    let x_offset = 300.0;
    let freq = -(2.0 * std::f64::consts::PI) / 10000.0;
    let stencil =
        nhls::standard_stencils::RotatingAdvectionStencil::new(freq, 0.1);

    // Create domains
    let mut input_domain = image_prob.input.as_slice_domain();
    let mut output_domain = image_prob.output.as_slice_domain();
    s_normal_ic_2d(&mut input_domain, x_offset, y_offset, args.chunk_size);

    let direct_solver = AP2DDirectSolver::new(&stencil);
    let planner_params = PlannerParameters {
        plan_type: args.plan_type,
        cutoff: args.cutoff,
        ratio: args.ratio,
        chunk_size: args.chunk_size,
    };
    let mut solver = TVAPSolver::new(
        &stencil,
        grid_bound,
        args.steps_per_image,
        args.threads,
        &planner_params,
        direct_solver,
    );

    if args.gen_only {
        args.save_wisdom();
        std::process::exit(0);
    }

    if args.write_images {
        image2d(&input_domain, &args.frame_name(0));
        write_csv_2d(&input_domain, &args.csv_frame_name(0));
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
            write_csv_2d(&input_domain, &args.csv_frame_name(t));
        }
    }

    args.save_wisdom();
}
