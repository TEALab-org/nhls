use core::f64;

use nhls::domain::*;
use nhls::image_1d_example::*;
use nhls::init::*;
use nhls::time_varying::*;
use nhls::util::*;

pub struct TVHeat1D {
    offsets: [Coord<1>; 3],
}

impl TVHeat1D {
    pub fn new() -> Self {
        let offsets = [vector![1], vector![-1], vector![0]];
        TVHeat1D { offsets }
    }
}

impl TVStencil<1, 3> for TVHeat1D {
    fn offsets(&self) -> &[Coord<1>; 3] {
        &self.offsets
    }

    fn weights(&self, global_time: usize) -> Values<3> {
        let t_f = global_time as f64;
        let e = (-t_f * 0.01).exp();
        let cw = 1.0 - e;
        let nw = e / 2.0;
        vector![nw, nw, cw]
    }
}

fn main() {
    let (args, output_image_path) = Args::cli_parse("tv_heat_1d_p_fft");

    // Grid size
    let grid_bound = args.grid_bounds();

    let stencil = TVHeat1D::new();

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
        args.save_wisdom();
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
        periodic_solver.apply(
            &mut input_domain,
            &mut output_domain,
            global_time,
        );
        global_time += args.steps_per_line;
        std::mem::swap(&mut input_domain, &mut output_domain);
        if let Some(i) = img.as_mut() {
            i.add_line(t as u32, input_domain.buffer());
        }
    }

    if let Some(i) = img {
        i.write(&output_image_path);
    }

    args.save_wisdom();
}
