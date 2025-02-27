use nhls::domain::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::init;
use nhls::solver::AP2DDirectSolver;
use nhls::stencil::*;

pub fn heat_2d(dt: f64, dx: f64, dy: f64, k_x: f64, k_y: f64) -> Stencil<2, 5> {
    Stencil::new(
        [[1, 0], [0, -1], [-1, 0], [0, 1], [0, 0]],
        move |args: &[f64; 5]| {
            let middle = args[4];
            let left = args[2];
            let right = args[0];
            let bottom = args[1];
            let top = args[3];
            middle
                + (k_x * dt / (dx * dx)) * (left - 2.0 * middle + right)
                + (k_y * dt / (dy * dy)) * (top - 2.0 * middle + bottom)
        },
    )
}

fn main() {
    let args = Args::cli_parse("heat_2d_ap_direct_specialized");

    // Grid size
    let grid_bound = args.grid_bounds();

    let stencil = heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);

    // Create domains
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);

    if args.rand_init {
        init::rand(&mut input_domain, 1024, args.chunk_size);
    } else {
        init::normal_ic_2d(&mut input_domain, args.chunk_size);
    }
    if args.write_images {
        image2d(&input_domain, &args.frame_name(0));
    }

    let solver = AP2DDirectSolver::new(
        &stencil,
        grid_bound,
        args.steps_per_image,
        args.threads,
    );

    // Apply direct solver
    //let mut global_time = 0;
    for t in 1..args.images {
        solver.apply(&mut input_domain, &mut output_domain);
        //global_time += args.steps_per_image;
        std::mem::swap(&mut input_domain, &mut output_domain);
        if args.write_images {
            image2d(&input_domain, &args.frame_name(t));
        }
    }
}
