use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::*;
use nhls::stencil::*;

fn main() {
    // Grid size
    let N: usize = 2000;

    //let final_t: usize = 200;

    let steps_per_image = 10;

    // Step size t
    let dt: f32 = 1.0;

    // Step size x
    let dx: f32 = 1.0;

    // Heat transfer coefficient
    let k: f32 = 0.3;

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
    let N_f = N as f32;
    let sigma_sq: f32 = (N_f / 25.0) * (N_f / 25.0);
    let ic_gen = |i: usize| {
        let x = (i as f32) - (N_f / 2.0);
        //let f = ( 1.0 / (2.0 * std::f32::consts::PI * sigma_sq)).sqrt();
        let exp = -x * x / (2.0 * sigma_sq);
        exp.exp()
    };
    for i in 0..N {
        grid_input[i] = ic_gen(i);
    }

    // Setup FFT stuff
    let mut forward_plan = fftw::plan::R2RPlan32::aligned(
        &[N],
        fftw::types::R2RKind::FFTW_R2HC,
        fftw::types::Flag::ESTIMATE,
    )
    .unwrap();

    let mut backward_plan = fftw::plan::R2RPlan32::aligned(
        &[N],
        fftw::types::R2RKind::FFTW_HC2R,
        fftw::types::Flag::ESTIMATE,
    )
    .unwrap();

    let mut fft_stencil_input_buffer = fftw::array::AlignedVec::new(N);
    let mut fft_stencil_output_buffer = fftw::array::AlignedVec::new(N);
    let mut fft_ic_input_buffer = fftw::array::AlignedVec::new(N);
    let mut fft_ic_output_buffer = fftw::array::AlignedVec::new(N);
    for i in 0..N {
        fft_stencil_input_buffer[i] = 0.0f32;
        fft_stencil_output_buffer[i] = 0.0f32;
        fft_ic_input_buffer[i] = 0.0f32;
        fft_ic_output_buffer[i] = 0.0f32;
    }

    // Forward FFT of U -> V
    let w = s.extract_weights();
    println!("weights: {:?}", w);
    for i in 0..w.len() {
        let index_raw = s.offsets()[i][0];
        let index = if index_raw < 0 {
            (N as i32 + index_raw) as usize
        } else {
            index_raw as usize
        };
        println!("index: {}, w: {}", index, w[i]);
        fft_stencil_input_buffer[index] = w[i];
    }

    forward_plan
        .r2r(
            &mut fft_stencil_input_buffer,
            &mut fft_stencil_output_buffer,
        )
        .unwrap();

    // Forward FFT of a0 ->
    for i in 0..N {
        fft_ic_input_buffer[i] = ic_gen(i);
    }
    forward_plan
        .r2r(&mut fft_ic_input_buffer, &mut fft_ic_output_buffer)
        .unwrap();

    // Repeated Square V
    for _ in 0..5 {
        for i in 0..N {
            let r = fft_stencil_output_buffer[i];
            fft_stencil_output_buffer[i] = r * r;
        }
    }

    let gradient = colorous::TURBO;
    let T = 2000;
    let mut test_img = image::RgbImage::new(N as u32, T);

    // Backward FFT of result V
    for t in 0..T {
        for i in 0..N {
            fft_ic_output_buffer[i] *= fft_stencil_output_buffer[i];
            fft_ic_input_buffer[i] = 0.0;
        }

        backward_plan
            .r2r(&mut fft_ic_output_buffer, &mut fft_ic_input_buffer)
            .unwrap();
        for i in 0..N as u32 {
            let c = gradient.eval_continuous((fft_ic_input_buffer[i as usize] / N as f32) as f64);
            test_img.put_pixel(i, t, image::Rgb(c.as_array()));
        }
    }

    test_img
        .save("test_image_fft.png")
        .expect("Couldn't save test img");
}
