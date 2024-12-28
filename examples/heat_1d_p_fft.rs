use nhls::domain::*;
use nhls::image_1d_example::*;
use nhls::util::*;

fn main() {
    let (args, output_image_path) = Args::cli_parse("heat_1d_p_fft");

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.5);

    // Create domains
    let grid_bound = args.grid_bounds();
    let mut input_domain = OwnedDomain::new(grid_bound);

    let mut output_domain = OwnedDomain::new(grid_bound);

    // Fill in with IC values (use normal dist for spike in the middle)
    let n_f = grid_bound.buffer_size() as f64;
    let sigma_sq: f64 = (n_f / 25.0) * (n_f / 25.0);
    let ic_gen = |world_coord: Coord<1>| {
        let x = (world_coord[0] as f64) - (n_f / 2.0);
        let exp = -x * x / (2.0 * sigma_sq);
        exp.exp()
    };
    input_domain.par_set_values(ic_gen, args.chunk_size);

    // Make image
    let mut img = nhls::image::Image1D::new(grid_bound, args.lines as u32);
    let mut periodic_library =
        nhls::solver::periodic_plan::PeriodicPlanLibrary::new(
            &grid_bound,
            &stencil,
        );
    img.add_line(0, input_domain.buffer());
    for t in 1..args.lines as u32 {
        periodic_library.apply(
            &mut input_domain,
            &mut output_domain,
            args.steps_per_line,
            args.chunk_size,
        );
        std::mem::swap(&mut input_domain, &mut output_domain);
        img.add_line(t, input_domain.buffer());
    }

    img.write(&output_image_path);
}
