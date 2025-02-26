use core::f64;

use nhls::domain::*;
use nhls::fft_solver::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::init::*;
use nhls::time_varying::*;
use nhls::util::*;
use std::time::*;

pub struct TVHeat2D {
    offsets: [Coord<2>; 5],
}

impl TVHeat2D {
    pub fn new() -> Self {
        let offsets = [
            vector![1, 0],
            vector![0, -1],
            vector![-1, 0],
            vector![0, 1],
            vector![0, 0],
        ];
        TVHeat2D { offsets }
    }
}

impl TVStencil<2, 5> for TVHeat2D {
    fn offsets(&self) -> &[Coord<2>; 5] {
        &self.offsets
    }

    fn weights(&self, global_time: usize) -> Values<5> {
        let t_f = global_time as f64;
        let e = (-t_f * 0.01).exp();
        let cw = 1.0 - e;
        let nw = e / 5.0;
        vector![nw, nw, nw, nw, cw]
    }
}

fn main() {
    let args = Args::cli_parse("tv_heat_2d_p_fft");

    // Grid size
    let grid_bound = args.grid_bounds();

    let stencil = TVHeat2D::new();

    // Create domains
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);
    rand(&mut input_domain, 10, args.chunk_size);

    //
    // PLAY WITH SOLVER SETUP
    //

    // AP TV Planner
    let planner_params = PlannerParameters {
        plan_type: args.plan_type,
        cutoff: args.cutoff,
        ratio: args.ratio,
        chunk_size: args.chunk_size,
    };
    let planner_result = create_tv_ap_plan(
        &stencil,
        grid_bound,
        args.steps_per_image,
        args.threads,
        &planner_params,
    );
    planner_result.plan.to_dot_file(&args.dot_path());
    /*
        planner_result
            .tree_query_collector
            .write_query_file(&args.query_file_path());
    */
    let mut tv_tree = TVTreePlanner::new(&stencil, grid_bound);
    tv_tree.build_range(0, args.steps_per_image, 0);
    tv_tree.to_dot_file(&args.tree_dot_path());

    // Tree planner

    // Are all the nodes in the tree?

    // So lets setup the planner, get that running again

    // Then make new tree that has hashtable for lookup as provided

    //
    // END SOLVER SETUP
    //

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
        /*
        solver.apply(
            &mut input_domain,
            &mut output_domain,
            global_time,
        );
        */
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
