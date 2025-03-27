use core::f64;

use nhls::domain::*;
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
    let args = Args::cli_setup("tv_heat_2d_p_fft");

    // Grid size
    let grid_bound = args.grid_bounds();

    let stencil = TVHeat2D::new();

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

    args.finish();
}
