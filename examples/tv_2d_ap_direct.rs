use core::f64;

use nhls::domain::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::init::*;
use nhls::standard_stencils::*;
use nhls::time_varying::*;
use nhls::util::*;

fn main() {
    let args = Args::cli_parse("tv_2d_ap_direct");

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
    // Create boundary condition
    let bc = ConstantCheck::new(0.0, grid_bound);

    // Apply direct solver
    let mut global_time = 0;
    for t in 1..args.images {
        tv_box_apply(
            &bc,
            &stencil,
            &mut input_domain,
            &mut output_domain,
            args.steps_per_image,
            global_time,
            args.chunk_size,
        );
        global_time += args.steps_per_image;
        std::mem::swap(&mut input_domain, &mut output_domain);
        if args.write_images {
            image2d(&input_domain, &args.frame_name(t));
        }
    }
}
