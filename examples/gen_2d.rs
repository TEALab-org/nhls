use nhls::domain::*;
use nhls::image_2d_example::*;
use nhls::solver::*;
use nhls::stencil::*;
use nhls::util::*;

fn main() {
    let args = Args::cli_parse("gen_2d");

    // Grid size
    let grid_bound = args.grid_bounds();

    let stencil = include!("gen_2d.stencil");

    // Create domains
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);

    // Fill in with IC values (use normal dist for spike in the middle)
    let width_f = grid_bound.bounds[(0, 1)] as f64 + 1.0;
    let height_f = grid_bound.bounds[(1, 1)] as f64 + 1.0;
    let sigma_sq: f64 = (width_f / 25.0) * (width_f / 25.0);
    let ic_gen = |coord: Coord<2>| {
        let x = (coord[0] as f64) - (width_f / 2.0);
        let y = (coord[1] as f64) - (height_f / 2.0);
        let r = (x * x + y * y).sqrt();
        let exp = -r * r / (2.0 * sigma_sq);
        exp.exp()
    };
    input_domain.par_set_values(ic_gen, args.chunk_size);

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
