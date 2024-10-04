use nhls::stencil::*;

fn grid_to_image(width: usize, height: usize, grid: &[f32], path: String) {
    let gradient = colorous::TURBO;
    let mut img = image::RgbImage::new(width as u32, height as u32);
    for l in 0..width * height {
        let (i, j) = linear_to_coord(width, height, l);
        let r = grid[l];
        let c = gradient.eval_continuous(r as f64);
        img.put_pixel(i as u32, j as u32, image::Rgb(c.as_array()));
    }
    img.save(&path).expect("Couldn't save image");
}

fn coord_to_linear(width: i32, i: i32, j: i32) -> usize {
    (i * width + j) as usize
}

fn linear_to_coord(width: usize, height: usize, index: usize) -> (i32, i32) {
    let i = (index / width) as i32;
    let j = (index % height) as i32;
    (i, j)
}

fn apply_2d_stencil_periodic<
    FloatType: num::Float + Clone,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
>(
    s: &Stencil<FloatType, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    input_slice: &[FloatType],
    i: i32,
    j: i32,
    width: i32,
    height: i32,
) -> FloatType
where
    Operation: StencilOperation<FloatType, NEIGHBORHOOD_SIZE>,
{
    let args = s.offsets().map(|offset| {
        let n_i_raw = (i + offset[0]) % width;
        let n_j_raw = (j + offset[1]) % height;

        let n_i = if n_i_raw < 0 {
            width + n_i_raw
        } else {
            n_i_raw
        };
        let n_j = if n_j_raw < 0 {
            height + n_j_raw
        } else {
            n_j_raw
        };

        let linear_index = coord_to_linear(width, n_i, n_j);
        /*
        println!(
            "s: ({}, {}), o: ({}, {}), r: ({}, {}), n: ({}, {}), l: {}",
            i, j, offset[0], offset[1], n_i_raw, n_j_raw, n_i, n_j, linear_index
        );
        */
        input_slice[linear_index]
    });
    s.apply(&args)
}

fn main() {
    // Grid size
    let width: usize = 1000;
    let height: usize = 1000;

    let steps_per_image = 64;

    let final_t: usize = 200 * steps_per_image;

    // Step size t
    let dt: f32 = 1.0;

    // Step size x
    let dx: f32 = 1.0;

    // Step size y
    let dy: f32 = 1.0;

    // Heat transfer coefficient
    let k_x: f32 = 0.1;
    let k_y: f32 = 0.1;

    let s = Stencil::new(
        [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]],
        |args: &[f32; 5]| {
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

    // Create buffers
    let mut grid_input = vec![0.0; width * height];
    let mut grid_output = vec![0.0; width * height];

    // Fill in with IC values (use normal dist for spike in the middle)
    let width_f = width as f32;
    let height_f = height as f32;
    let sigma_sq: f32 = (width_f / 25.0) * (width_f / 25.0);
    let ic_gen = |(i, j): (i32, i32)| {
        let x = (i as f32) - (width_f / 2.0);
        let y = (j as f32) - (height_f / 2.0);
        let r = (x * x + y * y).sqrt();
        //let f = ( 1.0 / (2.0 * std::f32::consts::PI * sigma_sq)).sqrt();
        let exp = -r * r / (2.0 * sigma_sq);
        exp.exp()
    };
    for l in 0..width * height {
        grid_input[l] = ic_gen(linear_to_coord(width, height, l));
    }

    grid_to_image(
        width,
        height,
        &grid_input,
        "heat_2d_naive/ic.png".to_owned(),
    );

    for t in 0..final_t + 1 {
        println!("t: {}", t);
        for l in 0..width * height {
            let (i, j) = linear_to_coord(width, height, l);
            let r = apply_2d_stencil_periodic(&s, &grid_input, i, j, width as i32, height as i32);
            grid_output[l] = r;
        }
        std::mem::swap(&mut grid_output, &mut grid_input);

        if t % steps_per_image == 0 {
            println!("Saving");
            grid_to_image(
                width,
                height,
                &grid_input,
                format!("heat_2d_naive/frame_{:04}.png", t / steps_per_image),
            );
        }
    }
}
