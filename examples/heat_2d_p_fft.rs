use nhls::domain::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::util::*;

fn main() {
    let args = Args::cli_parse("heat_2d_p_direct");

    let stencil = nhls::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);

    // Create domains
    let grid_bound = args.grid_bounds();
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);

    // Fill in with IC values (use normal dist for spike in the middle)
    // Write out initial frame
    let exclusive_bounds = grid_bound.exclusive_bounds();
    let width_f = exclusive_bounds[0] as f64;
    let height_f = exclusive_bounds[1] as f64;
    let sigma_sq: f64 = (width_f / 25.0) * (width_f / 25.0);
    let ic_gen = |coord: Coord<2>| {
        let x = (coord[0] as f64) - (width_f / 2.0);
        let y = (coord[1] as f64) - (height_f / 2.0);
        let r = (x * x + y * y).sqrt();
        let exp = -r * r / (2.0 * sigma_sq);
        exp.exp()
    };
    input_domain.par_set_values(ic_gen, args.chunk_size);
    if args.write_images {
        image2d(&input_domain, &args.frame_name(0));
    }

    // Apply periodic solver
    let mut periodic_library =
        nhls::solver::periodic_plan::PeriodicPlanLibrary::new(
            &grid_bound,
            &stencil,
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
}
