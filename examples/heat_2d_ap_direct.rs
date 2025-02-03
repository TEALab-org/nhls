use nhls::domain::*;
use nhls::image::*;
use nhls::image_2d_example::*;

fn main() {
    let args = Args::cli_parse("heat_2d_ap_direct");

    // Grid size
    let grid_bound = args.grid_bounds();

    let stencil = nhls::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);

    // Create domains
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);

    if args.write_images {
        image2d(&input_domain, &args.frame_name(0));
    }

    // Create boundary condition
    let bc = ConstantCheck::new(0.0, grid_bound);

    // Apply direct solver
    let mut global_time = 0;
    for t in 1..args.images {
        nhls::solver::direct::box_apply(
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
