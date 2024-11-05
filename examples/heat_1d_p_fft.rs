use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::*;
use nhls::stencil::*;

fn main() {
    // Grid size
    let N: usize = 16;

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
        0.25 * left + 0.5 * middle + 0.25 * right
    });

    // Fill in with IC values (use normal dist for spike in the middle)
    let N_f = N as f32;
    let sigma_sq: f32 = (N_f / 25.0) * (N_f / 25.0);
    let ic_gen = |i: usize| i as f32;

    // Setup FFT stuff
    let mut forward_plan =
        fftw::plan::R2CPlan32::aligned(&[N], fftw::types::Flag::ESTIMATE).unwrap();

    let mut backward_plan =
        fftw::plan::C2RPlan32::aligned(&[N], fftw::types::Flag::ESTIMATE).unwrap();

    let mut fft_stencil_input_buffer = fftw::array::AlignedVec::new(N);
    let mut fft_stencil_output_buffer = fftw::array::AlignedVec::new(N / 2 + 1);
    let mut fft_ic_input_buffer = fftw::array::AlignedVec::new(N);
    let mut fft_ic_output_buffer = fftw::array::AlignedVec::new(N / 2 + 1);
    for i in 0..N {
        fft_stencil_input_buffer[i] = 0.0f32;
        fft_ic_input_buffer[i] = 0.0f32;
    }

    for i in 0..N / 2 + 1 {
        fft_stencil_output_buffer[i] = c32::new(0.0f32, 0.0f32);
        fft_ic_output_buffer[i] = c32::new(0.0f32, 0.0f32);
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

    println!(
        "stencil_input_buffer: {:?}",
        fft_stencil_input_buffer.as_slice()
    );

    forward_plan
        .r2c(
            &mut fft_stencil_input_buffer,
            &mut fft_stencil_output_buffer,
        )
        .unwrap();

    println!(
        "stencil_output_buffer: {:?}",
        fft_stencil_output_buffer.as_slice()
    );

    // Forward FFT of a0 ->
    for i in 0..N {
        fft_ic_input_buffer[i] = ic_gen(i);
    }

    println!("ic input: {:?}", fft_ic_input_buffer.as_slice());
    forward_plan
        .r2c(&mut fft_ic_input_buffer, &mut fft_ic_output_buffer)
        .unwrap();
    println!("ic output: {:?}", fft_ic_output_buffer.as_slice());

    // Repeated Square V
    for _ in 0..1 {
        for i in 0..N / 2 + 1 {
            let r = fft_stencil_output_buffer[i];
            fft_stencil_output_buffer[i] = r * r;
        }
    }
    println!(
        "repeated squares: {:?}",
        fft_stencil_output_buffer.as_slice()
    );

    let gradient = colorous::TURBO;
    //let T = 2000;
    //let mut test_img = image::RgbImage::new(N as u32, T);

    // Backward FFT of result V
    for t in 0..1 {
        for i in 0..N / 2 + 1 {
            fft_ic_output_buffer[i] *= fft_stencil_output_buffer[i];
        }

        for i in 0..N {
            fft_ic_input_buffer[i] = 0.0;
        }

        println!("y: {:?}", fft_ic_output_buffer.as_slice());

        backward_plan
            .c2r(&mut fft_ic_output_buffer, &mut fft_ic_input_buffer)
            .unwrap();

        for i in 0..N {
            fft_ic_input_buffer[i] /= 16.0;
        }
        /*
        for i in 0..N as u32 {
            let c = gradient.eval_continuous((fft_ic_input_buffer[i as usize] / N as f32) as f64);
            test_img.put_pixel(i, t, image::Rgb(c.as_array()));
        }
        */
        println!("output: {:?}", fft_ic_input_buffer.as_slice());
    }
    /*
        test_img
            .save("test_image_fft.png")
            .expect("Couldn't save test img");
    */
    //fftw::wisdom::export_wisdom_file_f32(&"/tmp/wisdom_f32").unwrap();
    //fftw::wisdom::export_wisdom_file_f64(&"/tmp/wisdom_f64").unwrap();
}
