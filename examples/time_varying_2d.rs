use nhls::domain::*;
use nhls::fft_solver::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::stencil::*;
use nhls::util::*;

pub struct PulseBC<const GRID_DIMENSION: usize> {
    rate: f64,
    n_f: f64,
    sigma_sq: f64,
    aabb: AABB<GRID_DIMENSION>,
}

impl<const GRID_DIMENSION: usize> PulseBC<GRID_DIMENSION> {
    pub fn new(rate: f64, aabb: AABB<GRID_DIMENSION>) -> Self {
        let n_f = aabb.exclusive_bounds()[1] as f64;
        let sigma_sq: f64 = (n_f / 10.0) * (n_f / 10.0);
        PulseBC {
            rate,
            aabb,
            n_f,
            sigma_sq,
        }
    }
}

impl<const GRID_DIMENSION: usize> BCCheck<GRID_DIMENSION>
    for PulseBC<GRID_DIMENSION>
{
    fn check(
        &self,
        coord: &Coord<GRID_DIMENSION>,
        global_time: usize,
    ) -> Option<f64> {
        if self.aabb.contains(coord) {
            None
        } else if coord[0] < self.aabb.bounds[(0, 0)] {
            let x = (coord[1] as f64) - (self.n_f / 2.0);
            let exp = -x * x / (2.0 * self.sigma_sq);
            let normal_value = exp.exp();
            let sin_mod = 0.5 * ((global_time as f64 * self.rate).sin() + 1.0);
            Some(sin_mod * normal_value)
        } else {
            Some(0.0)
        }
    }
}

fn main() {
    let args = Args::cli_parse("time_varying_2d");

    let stencil = Stencil::new(
        [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]],
        move |args: &[f64; 5]| {
            let middle = args[0];
            let left = args[1];
            let right = args[2];
            let bottom = args[3];
            let top = args[4];
            0.1 * middle + 0.4 * left + 0.1 * right + 0.2 * top + 0.2 * bottom
        },
    );

    let w = stencil.extract_weights();
    let s: f64 = w.iter().sum();
    println!("{:?}, {}", w, s);
    let grid_bound = args.grid_bounds();

    // Create BC
    let bc = PulseBC::new((2.0 * std::f64::consts::PI) / 1000.0, grid_bound);

    // Create AP Solver
    let planner_params = PlannerParameters {
        plan_type: args.plan_type,
        cutoff: 40,
        ratio: 0.5,
        chunk_size: args.chunk_size,
    };
    let solver = APSolver::new(
        &bc,
        &stencil,
        grid_bound,
        args.steps_per_image,
        &planner_params,
    );
    if args.write_dot {
        let mut dot_path = args.output_dir.clone();
        dot_path.push("plan.dot");
        solver.to_dot_file(&dot_path);

        let mut d_path = args.output_dir.clone();
        d_path.push("scratch.txt");
        solver.scratch_descriptor_file(&d_path);
    }

    // Create domains
    let mut buffer_1 = OwnedDomain::new(grid_bound);
    let mut buffer_2 = OwnedDomain::new(grid_bound);
    let mut input_domain = buffer_1.as_slice_domain();
    let mut output_domain = buffer_2.as_slice_domain();
    if args.write_images {
        image2d(&input_domain, &args.frame_name(0));
    }

    let mut global_time = 0;
    for t in 1..args.images {
        solver.apply(&mut input_domain, &mut output_domain, global_time);
        global_time += args.steps_per_image;
        std::mem::swap(&mut input_domain, &mut output_domain);
        if args.write_images {
            image2d(&input_domain, &args.frame_name(t));
        }
    }

    args.save_wisdom();
}
