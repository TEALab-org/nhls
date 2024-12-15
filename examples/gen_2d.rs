use nalgebra::matrix;
use nhls::domain::*;
use nhls::solver::*;
use nhls::stencil::*;
use nhls::util::*;
use rayon::prelude::*;

fn main() {
    const GRID_DIMENSION: usize = 2;

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
    let mut output_domain: Domain<2> = Domain::new(grid_bound, &mut grid_output);

    // Fill in with IC values (use normal dist for spike in the middle)
    let width_f = grid_bound.bounds[(0, 1)] as f32 + 1.0;
    let height_f = grid_bound.bounds[(1, 1)] as f32 + 1.0;
    let sigma_sq: f32 = (width_f / 25.0) * (width_f / 25.0);
    input_domain
        .par_modify_access(100)
        .for_each(|mut d: DomainChunk<'_, GRID_DIMENSION>| {
            d.coord_iter_mut().for_each(
                |(world_coord, value_mut): (Coord<GRID_DIMENSION>, &mut f32)| {
                    let x = (world_coord[0] as f32) - (width_f / 2.0);
                    let y = (world_coord[1] as f32) - (height_f / 2.0);
                    let r = (x * x + y * y).sqrt();
                    let exp = -r * r / (2.0 * sigma_sq);
                    *value_mut = exp.exp()
                },
            )
        });

    // Make image
    nhls::image::image2d(&input_domain, "gen_2d/frame_000.png");
    for t in 1..n_images as u32 {
        periodic_naive::box_solve(
            &stencil,
            &mut input_domain,
            &mut output_domain,
            n_steps_per_image,
            chunk_size,
        );
        std::mem::swap(&mut input_domain, &mut output_domain);
        nhls::image::image2d(&input_domain, &format!("gen_2d/frame_{:03}.png", t));
    }
}
