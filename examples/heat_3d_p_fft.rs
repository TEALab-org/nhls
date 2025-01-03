use nhls::domain::*;
use nhls::image_3d_example::*;
use nhls::init;
use nhls::vtk::*;

fn main() {
    let args = Args::cli_parse("heat_3d_p_fft");

    let stencil =
        nhls::standard_stencils::heat_3d(1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1);

    // Create domains
    let grid_bound = args.grid_bounds();
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);

    if args.rand_init {
        init::rand(&mut input_domain, 1024, args.chunk_size);
    } else {
        init::normal_ic_3d(&mut input_domain, args.chunk_size);
    }
    write_vtk3d(&input_domain, &args.frame_name(0));

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
            write_vtk3d(&input_domain, &args.frame_name(t));
        }
    }

    args.save_wisdom();
}
