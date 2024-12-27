use nhls::domain::*;
use nhls::solver::*;
use nhls::stencil::*;
use nhls::util::*;

mod util;

fn main() {
    let args = util::Args::cli_parse("gen_1d");
    let mut output_image_path = args.output_dir.clone();
    output_image_path.push("gen_1d.png");

    // Grid size
    let grid_bound = AABB::new(matrix![0, 999]);

    let n_lines = 1000;

    let n_steps_per_line = 16;

    let chunk_size = 100;

    let stencil = include!("gen_1d.stencil");

    // Create domains
    let buffer_size = grid_bound.buffer_size();
    let mut grid_input = vec![0.0; buffer_size];
    let mut input_domain = Domain::new(grid_bound, &mut grid_input);

    let mut grid_output = vec![0.0; buffer_size];
    let mut output_domain = Domain::new(grid_bound, &mut grid_output);

    // Fill in with IC values (use normal dist for spike in the middle)
    let n_f = buffer_size as f64;
    let sigma_sq: f64 = (n_f / 25.0) * (n_f / 25.0);
    let ic_gen = |coord: Coord<1>| {
        let x = (coord[0] as f64) - (n_f / 2.0);
        //let f = ( 1.0 / (2.0 * std::f64::consts::PI * sigma_sq)).sqrt();
        let exp = -x * x / (2.0 * sigma_sq);
        exp.exp()
    };
    input_domain.par_set_values(ic_gen, chunk_size);

    // Make image
    let mut img = nhls::image::Image1D::new(grid_bound, n_lines as u32);
    img.add_line(0, input_domain.buffer());
    for t in 1..n_lines as u32 {
        direct_periodic_apply(
            &stencil,
            &mut input_domain,
            &mut output_domain,
            n_steps_per_line,
            chunk_size,
        );
        std::mem::swap(&mut input_domain, &mut output_domain);
        img.add_line(t, input_domain.buffer());
    }

    img.write(&output_image_path);
}
