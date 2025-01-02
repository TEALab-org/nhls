use nhls::domain::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::init;

fn main() {
    let args = Args::cli_parse("heat_2d_p_fft");

    let stencil = nhls::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);

    // Create domains
    let grid_bound = args.grid_bounds();
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);

    if args.rand_init {
        init::rand(&mut input_domain, 1024, args.chunk_size);
    } else {
        init::normal_ic_2d(&mut input_domain, args.chunk_size);
    }

    // Apply periodic solver
    let mut periodic_library =
        nhls::solver::periodic_plan::PeriodicPlanLibrary::new(
            &grid_bound,
            &stencil,
            args.plan_type,
        );
    for t in 1..args.images {
        periodic_library.apply(
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

    args.save_wisdom();
}
