use nhls::domain::*;
use nhls::image_1d_example::*;
use nhls::init;
use nhls::solver::*;
use nhls::stencil::*;

fn main() {
    let (args, output_image_path) = Args::cli_parse("gen_1d");

    let stencil = include!("gen_1d.stencil");

    // Create domains
    let grid_bound = args.grid_bounds();
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);

    if args.rand_init {
        init::rand(&mut input_domain, 1024, args.chunk_size);
    } else {
        init::normal_ic_1d(&mut input_domain, args.chunk_size);
    }

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
