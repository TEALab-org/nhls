use fftw::array::AlignedVec;
use nhls::domain::*;
use nhls::image::*;
use nhls::stencil::*;
use nhls::util::*;

mod util;

fn main() {
    let args = util::Args::cli_parse("heat_2d_p_direct");

    // Grid size
    let grid_bound = AABB::new(matrix![0, 999; 0, 999]);

    let steps_per_image = 64;

    let n_images = 200;

    let chunk_size = 1000;

    // Step size t
    let dt: f64 = 1.0;

    // Step size x
    let dx: f64 = 1.0;

    // Step size y
    let dy: f64 = 1.0;

    // Heat transfer coefficient
    let k_x: f64 = 0.2;
    let k_y: f64 = 0.2;

    let stencil = Stencil::new(
        [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]],
        |args: &[f64; 5]| {
            let middle = args[0];
            let left = args[1];
            let right = args[2];
            let bottom = args[3];
            let top = args[4];
            middle
                + (k_x * dt / (dx * dx)) * (left - 2.0 * middle + right)
                + (k_y * dt / (dy * dy)) * (top - 2.0 * middle + bottom)
        },
    );

    // Create domains
    let buffer_size = grid_bound.buffer_size();
    let mut input_buffer = AlignedVec::new(buffer_size);
    let mut output_buffer = AlignedVec::new(buffer_size);
    let mut input_domain = Domain::new(grid_bound, &mut input_buffer);
    let mut output_domain = Domain::new(grid_bound, &mut output_buffer);

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

    // Create boundary condition
    let bc = ConstantCheck::new(0.0, grid_bound);

    // Apply direct solver
    for t in 1..n_images {
        nhls::solver::direct::box_apply(
            &bc,
            &stencil,
            &mut input_domain,
            &mut output_domain,
            steps_per_image,
            chunk_size,
        );

        std::mem::swap(&mut input_domain, &mut output_domain);
        image2d(&input_domain, &args.frame_name(t));
    }
}
