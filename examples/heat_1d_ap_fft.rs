use nhls::ap_solver::*;
use nhls::domain::*;
use nhls::image_1d_example::*;
use std::time::*;

fn main() {
    let (args, output_image_path) = Args::cli_setup("heat_1d_ap_fft");

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.5);
    let grid_bound = args.grid_bounds();

    // Create domains
    let mut buffer_1 = OwnedDomain::new(grid_bound);
    let mut buffer_2 = OwnedDomain::new(grid_bound);
    let mut input_domain = buffer_1.as_slice_domain();
    let mut output_domain = buffer_2.as_slice_domain();

    // This optimized direct solver implement a uniform boundary condition of 0.0
    let direct_solver = DirectSolver3Pt1DOpt::new(&stencil);

    // Create AP Solver
    let solver_params = args.solver_parameters();
    let mut solver =
        generate_ap_solver(&stencil, direct_solver, &solver_params);

    solver.print_report();

    if args.write_dot {
        solver.to_dot_file(&args.dot_path());
    }

    if args.gen_only {
        args.finish();
        std::process::exit(0);
    }

    let mut img = None;
    if args.write_image {
        let mut i = nhls::image::Image1D::new(grid_bound, args.lines as u32);
        i.add_line(0, input_domain.buffer());
        img = Some(i);
    }

    let mut global_time = 0;
    for t in 1..args.lines as u32 {
        let now = Instant::now();
        solver.apply(&mut input_domain, &mut output_domain, global_time);
        let elapsed_time = now.elapsed();
        eprintln!("{}", elapsed_time.as_nanos() as f64 / 1000000000.0);

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
