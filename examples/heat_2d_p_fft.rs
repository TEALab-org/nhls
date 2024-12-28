use nhls::domain::*;
use nhls::image::*;
use nhls::util::*;

mod util;

fn main() {
    let args = util::Args::cli_parse("heat_2d_p_direct");

    // Grid size
    let grid_bound = AABB::new(matrix![0, 999; 0, 999]);

    let steps_per_image = 64;

    let n_images = 200;

    let chunk_size = 1000;

    let stencil = nhls::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);

    // Create domains
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
    input_domain.par_set_values(ic_gen, chunk_size);
    image2d(&input_domain, &args.frame_name(0));

    // Apply periodic solver
    let mut periodic_library =
        nhls::solver::periodic_plan::PeriodicPlanLibrary::new(
            &grid_bound,
            &stencil,
        );
    for t in 1..n_images {
        periodic_library.apply(
            &mut input_domain,
            &mut output_domain,
            steps_per_image,
            chunk_size,
        );
        std::mem::swap(&mut input_domain, &mut output_domain);
        image2d(&input_domain, &args.frame_name(t));
    }
}
