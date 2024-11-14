use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::*;
use nhls::par_slice;
use nhls::stencil::*;
use num_traits::identities::Zero;

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

fn linear_to_coord(width: usize, _: usize, index: usize) -> (i32, i32) {
    let i = (index / width) as i32;
    let j = (index % width) as i32;
    (i, j)
}

fn main() {
    // Grid size
    let width: usize = 1000;
    let height: usize = 1000;

    let steps_per_image = 64;

    let final_t: usize = 200 * steps_per_image;

    let chunk_size = 1000;

    // Step size t
    let dt: f32 = 1.0;

    // Step size x
    let dx: f32 = 1.0;

    // Step size y
    let dy: f32 = 1.0;

    // Heat transfer coefficient
    let k_x: f32 = 0.2;
    let k_y: f32 = 0.2;

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

    // Setup FFT stuff
    let mut forward_plan =
        fftw::plan::R2CPlan32::aligned(&[width, height], fftw::types::Flag::ESTIMATE).unwrap();

    let mut backward_plan =
        fftw::plan::C2RPlan32::aligned(&[width, height], fftw::types::Flag::ESTIMATE).unwrap();

    let mut fft_stencil_input_buffer = fftw::array::AlignedVec::new(width * height);
    let mut fft_stencil_output_buffer = fftw::array::AlignedVec::new(width * (height / 2 + 1));
    let mut squared_stencil_buffer = fftw::array::AlignedVec::new(width * (height / 2 + 1));
    let mut fft_backwards_buffer = fftw::array::AlignedVec::new(width * (height / 2 + 1));
    let mut fft_ic_input_buffer = fftw::array::AlignedVec::new(width * height);
    let mut fft_ic_output_buffer = fftw::array::AlignedVec::new(width * (height / 2 + 1));
    for i in 0..width * height {
        fft_stencil_input_buffer[i] = 0.0f32;
        fft_ic_input_buffer[i] = 0.0f32;
    }

    for i in 0..width * (height / 2 + 1) {
        fft_stencil_output_buffer[i] = c32::new(0.0, 0.0);
        fft_ic_output_buffer[i] = c32::new(0.0, 0.0);
        fft_backwards_buffer[i] = c32::new(0.0, 0.0);
        squared_stencil_buffer[i] = c32::new(0.0, 0.0);
    }

    // Forward FFT of U -> V
    let w = s.extract_weights();
    println!("weights: {:?}", w);
    for w_i in 0..w.len() {
        let i_raw = s.offsets()[w_i][0];
        let j_raw = s.offsets()[w_i][1];
        let i = if i_raw < 0 {
            width as i32 + i_raw
        } else {
            i_raw
        };
        let j = if j_raw < 0 {
            height as i32 + j_raw
        } else {
            j_raw
        };
        /*
        let i = i_raw + 500;
        let j = j_raw + 500;
        */
        let l = coord_to_linear(width as i32, i, j);
        println!("coord: ({}, {}), l: {}, w: {}", i, j, l, w[w_i]);
        fft_stencil_input_buffer[l] = w[w_i];
    }

    forward_plan
        .r2c(
            &mut fft_stencil_input_buffer,
            &mut fft_stencil_output_buffer,
        )
        .unwrap();

    // Forward FFT of a0 ->
    for l in 0..width * height {
        fft_ic_input_buffer[l] = ic_gen(linear_to_coord(width, height, l));
    }

    forward_plan
        .r2c(&mut fft_ic_input_buffer, &mut fft_ic_output_buffer)
        .unwrap();

    par_slice::power(
        steps_per_image,
        fft_stencil_output_buffer.as_slice_mut(),
        squared_stencil_buffer.as_slice_mut(),
        chunk_size,
    );

    // Backward FFT of result V
    for t in 0..200 {
        par_slice::multiply_by(
            fft_ic_output_buffer.as_slice_mut(),
            squared_stencil_buffer.as_slice_mut(),
            chunk_size,
        );
        par_slice::copy(
            fft_backwards_buffer.as_slice_mut(),
            fft_ic_output_buffer.as_slice(),
            chunk_size,
        );

        backward_plan
            .c2r(&mut fft_backwards_buffer, &mut fft_ic_input_buffer)
            .unwrap();

        par_slice::div(
            fft_ic_input_buffer.as_slice_mut(),
            (width * height) as f32,
            chunk_size,
        );

        grid_to_image(
            width,
            height,
            fft_ic_input_buffer.as_slice(),
            format!("heat_2d_fft/frame_{:04}.png", t),
        );
    }
}
