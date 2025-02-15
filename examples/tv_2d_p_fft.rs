use core::f64;

use nhls::domain::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::init::*;
use nhls::time_varying::*;
use nhls::util::*;

pub struct RotatingAdvectionStencil {
    offsets: [Coord<2>; 5],

    /// Steps per rotation
    frequency: f64,

    /// Hold central weight constant
    central_weight: f64,

    dist_mod: f64,
}

impl RotatingAdvectionStencil {
    pub fn new(frequency: f64, central_weight: f64) -> Self {
        let offsets = [
            vector![1, 0],
            vector![0, -1],
            vector![-1, 0],
            vector![0, 1],
            vector![0, 0],
        ];
        let dist_mod = 1.0 - central_weight;
        RotatingAdvectionStencil {
            offsets,
            frequency,
            central_weight,
            dist_mod,
        }
    }
}

impl TVStencil<2, 5> for RotatingAdvectionStencil {
    fn offsets(&self) -> &[Coord<2>; 5] {
        &self.offsets
    }

    // Model advection distribution with
    // *(0.5 sin(a + time * frequency) + 1.0) / 2Pi
    // That integrates to 1.0 over unit circle.
    // So each of the neighbors gets a quadrant,
    // i.e. integrate that for a in (0, Pi / 2) for quadrant 1
    // Did that with sympy
    // and got these equations
    fn weights(&self, global_time: usize) -> Values<5> {
        let f_gt = global_time as f64;
        let sqrt_2 = 2.0f64.sqrt();
        let pi = f64::consts::PI;
        let q1 = self.dist_mod
            * (sqrt_2 * (f_gt * self.frequency + pi / 4.0).sin() + pi)
            / (4.0 * pi);
        let q2 = self.dist_mod
            * (sqrt_2 * (f_gt * self.frequency + pi / 4.0).cos() + pi)
            / (4.0 * pi);
        let q3 = self.dist_mod
            * (-sqrt_2 * (f_gt * self.frequency + pi / 4.0).sin() + pi)
            / (4.0 * pi);
        let q4 = self.dist_mod
            * (-sqrt_2 * (f_gt * self.frequency + pi / 4.0).cos() + pi)
            / (4.0 * pi);

        vector![q1, q2, q3, q4, self.central_weight]
    }
}

fn main() {
    let args = Args::cli_parse("tv_2d_p_fft");

    // Grid size
    let grid_bound = args.grid_bounds();

    let freq = (2.0 * f64::consts::PI) / 1500.0;
    let stencil = RotatingAdvectionStencil::new(freq, 0.1);

    // Create domains
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);
    normal_ic_2d(&mut input_domain, args.chunk_size);

    if args.write_images {
        image2d(&input_domain, &args.frame_name(0));
    }

    let mut periodic_solver = TVPeriodicSolver::new(
        &stencil,
        args.steps_per_image,
        args.plan_type,
        grid_bound,
        args.threads,
        args.chunk_size,
    );

    // Apply direct solver
    let mut global_time = 0;
    for t in 1..args.images {
        periodic_solver.apply(
            &mut input_domain,
            &mut output_domain,
            global_time,
        );
        global_time += args.steps_per_image;
        std::mem::swap(&mut input_domain, &mut output_domain);
        if args.write_images {
            image2d(&input_domain, &args.frame_name(t));
        }
    }
}
