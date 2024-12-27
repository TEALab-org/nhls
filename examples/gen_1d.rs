use nhls::domain::*;
use nhls::image_1d_example::*;
use nhls::solver::*;
use nhls::stencil::*;
use nhls::util::*;

fn main() {
    let (args, output_image_path) = Args::cli_parse("gen_1d");

    let stencil = include!("gen_1d.stencil");

    // Create domains
    let grid_bound = args.grid_bounds();
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
    input_domain.par_set_values(ic_gen, args.chunk_size);

    // Make image
    let mut img = nhls::image::Image1D::new(grid_bound, args.lines as u32);
    img.add_line(0, input_domain.buffer());
    for t in 1..args.lines as u32 {
        direct_periodic_apply(
            &stencil,
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
