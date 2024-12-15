use nhls::domain::*;
use nhls::solver::*;
use nhls::stencil::*;
use nhls::util::*;
use rayon::prelude::*;

use nalgebra::matrix;

fn main() {
    const GRID_DIMENSION: usize = 1;

    // Grid size
    let grid_bound = AABB::new(matrix![0, 999]);

    let n_lines = 1000;

    let n_steps_per_line = 16;

    // Step size t
    let dt: f32 = 1.0;

    // Step size x
    let dx: f32 = 1.0;

    // Heat transfer coefficient
    let k: f32 = 0.5;

    let chunk_size = 100;

    let stencil = Stencil::new([[-1], [0], [1]], |args: &[f32; 3]| {
        let left = args[0];
        let middle = args[1];
        let right = args[2];
        middle + (k * dt / (dx * dx)) * (left - 2.0 * middle + right)
    });

    // Create domains
    let buffer_size = grid_bound.buffer_size();
    let mut grid_input = vec![0.0; buffer_size];
    let mut input_domain = Domain::new(grid_bound, &mut grid_input);

    let mut grid_output = vec![0.0; buffer_size];
    let mut output_domain = Domain::new(grid_bound, &mut grid_output);

    // Fill in with IC values (use normal dist for spike in the middle)
    let n_f = buffer_size as f32;
    let sigma_sq: f32 = (n_f / 25.0) * (n_f / 25.0);
    input_domain
        .par_modify_access(100)
        .for_each(|mut d: DomainChunk<'_, GRID_DIMENSION>| {
            d.coord_iter_mut().for_each(
                |(world_coord, value_mut): (Coord<GRID_DIMENSION>, &mut f32)| {
                    let x = (world_coord[0] as f32) - (n_f / 2.0);
                    //let f = ( 1.0 / (2.0 * std::f32::consts::PI * sigma_sq)).sqrt();
                    let exp = -x * x / (2.0 * sigma_sq);
                    *value_mut = exp.exp()
                },
            )
        });

    // Make image
    let mut img = nhls::image::Image1D::new(grid_bound, n_lines as u32);
    img.add_line(0, input_domain.buffer());
    for t in 1..n_lines as u32 {
        periodic_naive::box_solve(
            &stencil,
            &mut input_domain,
            &mut output_domain,
            n_steps_per_line,
            chunk_size,
        );
        std::mem::swap(&mut input_domain, &mut output_domain);
        img.add_line(t, input_domain.buffer());
    }

    img.write("test_image_01.png");
}
