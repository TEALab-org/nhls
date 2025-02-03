use nhls::domain::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::init;

fn main() {
    let args = Args::cli_parse("heat_2d_ap_direct");

    // Grid size
    let grid_bound = args.grid_bounds();

    let stencil = nhls::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);

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

    // Create boundary condition
    let bc = ConstantCheck::new(0.0, grid_bound);

    // Apply direct solver
    for t in 1..args.images {
        nhls::solver::direct::box_apply(
            &bc,
            &stencil,
            &mut input_domain,
            &mut output_domain,
            args.steps_per_image,
            args.chunk_size,
        );

        std::mem::swap(&mut input_domain, &mut output_domain);
        if args.write_images {
            image2d(&input_domain, &args.frame_name(t));
        }
    }
}
