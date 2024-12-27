use nhls::domain::*;
use nhls::solver::*;
use nhls::stencil::*;
use nhls::util::*;
mod util;

fn main() {
    let args = util::Args::cli_parse("gen_2d");

    // Grid size
    let grid_bound = AABB::new(matrix![0, 999; 0, 999]);

    let n_images = 40;

    let n_steps_per_image = 9;

    let stencil = include!("gen_2d.stencil");

    let chunk_size = 1000;

    // Create domains
    let buffer_size = grid_bound.buffer_size();
    let mut grid_input = vec![0.0; buffer_size];
    let mut input_domain: Domain<2> = Domain::new(grid_bound, &mut grid_input);

    let mut grid_output = vec![0.0; buffer_size];
    let mut output_domain: Domain<2> =
        Domain::new(grid_bound, &mut grid_output);

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
    input_domain.par_set_values(ic_gen, chunk_size);

    // Make image
    nhls::image::image2d(&input_domain, &args.frame_name(0));
    for t in 1..n_images as u32 {
        direct_periodic_apply(
            &stencil,
            &mut input_domain,
            &mut output_domain,
            n_steps_per_image,
            chunk_size,
        );
        std::mem::swap(&mut input_domain, &mut output_domain);
        nhls::image::image2d(&input_domain, &args.frame_name(t));
    }
}
