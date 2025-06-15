/*
use nhls::domain::*;
use nhls::image_1d_example::*;
use nhls::solver::*;

fn main() {
    let args = Args::cli_setup("heat_1d_ap_direct");

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.5);

    // Create BC
    let bc = ConstantCheck::new(1.0, grid_bound);

    // Make image
    let mut img = None;
    if args.write_image {
        let mut i = nhls::image::Image1D::new(grid_bound, args.lines as u32);
        i.add_line(0, input_domain.buffer());
        img = Some(i);
    }

    let mut global_time = 0;
    for t in 1..args.lines as u32 {
        box_apply(
            &bc,
            &stencil,
            &mut input_domain,
            &mut output_domain,
            args.steps_per_line,
            global_time,
            args.chunk_size,
        );
        global_time += args.steps_per_line;
        std::mem::swap(&mut input_domain, &mut output_domain);
        if let Some(i) = img.as_mut() {
            i.add_line(t, input_domain.buffer());
        }
    }
    if let Some(i) = img {
        i.write(&output_image_path.unwrap());
    }

    args.finish();
}
*/
