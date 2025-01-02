use nhls::domain::*;
use nhls::image_2d_example::*;
use nhls::init;
use nhls::solver::*;
use nhls::stencil::*;

fn main() {
    let args = Args::cli_parse("gen_2d");

    // Grid size
    let grid_bound = args.grid_bounds();

    let stencil = include!("gen_2d.stencil");

    // Create domains
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);

    if args.rand_init {
        init::rand(&mut input_domain, 1024, args.chunk_size);
    } else {
        init::normal_ic_2d(&mut input_domain, args.chunk_size);
    }

    // Make image
    nhls::image::image2d(&input_domain, &args.frame_name(0));
    for t in 1..args.images {
        direct_periodic_apply(
            &stencil,
            &mut input_domain,
            &mut output_domain,
            args.steps_per_image,
            args.chunk_size,
        );
        std::mem::swap(&mut input_domain, &mut output_domain);
        nhls::image::image2d(&input_domain, &args.frame_name(t));
    }
}
