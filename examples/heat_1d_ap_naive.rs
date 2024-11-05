use nhls::stencil::*;

fn apply_1d_stencil<
    FloatType: num::Float + Clone,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
>(
    s: &Stencil<FloatType, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    input_slice: &[FloatType],
    index: i32,
) -> FloatType
where
    Operation: StencilOperation<FloatType, NEIGHBORHOOD_SIZE>,
{
    let args = s.offsets().map(|offset| {
        let n_index = (index + offset[0]) as usize;
        input_slice[n_index]
    });
    s.apply(&args)
}

fn main() {
    // Grid size
    let N: usize = 2000;

    let final_t: usize = 2000 * 32;

    let steps_per_image = 16;

    // Step size t
    let dt: f32 = 1.0;

    // Step size x
    let dx: f32 = 1.0;

    // Heat transfer coefficient
    let k: f32 = 0.5;

    let s = Stencil::new([[-1], [0], [1]], |args: &[f32; 3]| {
        let left = args[0];
        let middle = args[1];
        let right = args[2];
        middle + (k * dt / (dx * dx)) * (left - 2.0 * middle + right)
    });

    // Create buffers
    let mut grid_input = vec![0.0; N];
    let mut grid_output = vec![0.0; N];

    // Fill in with IC values (use normal dist for spike in the middle)
    let ic_gen = |_: usize| 0.0;
    for i in 1..N {
        grid_input[i] = ic_gen(i);
    }
    grid_input[0] = 1.0;
    grid_input[N - 1] = 1.0;

    // Make image
    let gradient = colorous::TURBO;
    let mut test_img = image::RgbImage::new(N as u32, (final_t / steps_per_image) as u32);
    for t in 0..final_t as u32 {
        //println!("t: {}", t);
        for x in 1..N - 1 {
            let r = apply_1d_stencil(&s, &grid_input, x as i32);
            grid_output[x] = r;

            if t as usize % steps_per_image == 0 {
                let c = gradient.eval_continuous(r as f64);
                let y = t / steps_per_image as u32;
                test_img.put_pixel(x as u32, y, image::Rgb(c.as_array()));
            }
        }
        std::mem::swap(&mut grid_output, &mut grid_input);
    }

    test_img
        .save("heat_1d_ap_naive.png")
        .expect("Couldn't save test img");
}
