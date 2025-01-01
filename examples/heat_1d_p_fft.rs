use nhls::domain::*;
use nhls::image_1d_example::*;
use nhls::util::*;

fn main() {
    let (args, output_image_path) = Args::cli_parse("heat_1d_p_fft");

    fftw::threading::init_threads_f64().unwrap();
    fftw::threading::plan_with_nthreads_f64(8);

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

    let mut periodic_library =
        nhls::solver::periodic_plan::PeriodicPlanLibrary::new(
            &grid_bound,
            &stencil,
        );

    let mut img = None;
    if args.write_image {
        let mut i = nhls::image::Image1D::new(grid_bound, args.lines as u32);
        i.add_line(0, input_domain.buffer());
        img = Some(i);
    }

    for t in 1..args.lines as u32 {
        periodic_library.apply(
            &mut input_domain,
            &mut output_domain,
            args.steps_per_line,
            args.chunk_size,
        );
        std::mem::swap(&mut input_domain, &mut output_domain);
        if let Some(i) = img.as_mut() {
            i.add_line(t, input_domain.buffer());
        }
    }
    if let Some(i) = img {
        i.write(&output_image_path);
    }
}
