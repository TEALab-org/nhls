use nhls::ap_solver::*;
use nhls::domain::*;
use nhls::fft_solver::DirectFrustrumSolver;
use nhls::image_3d_example::*;
use nhls::vtk::*;
use std::time::*;

fn main() {
    let args = Args::cli_setup("heat_3d_ap_fft");

    let stencil =
        nhls::standard_stencils::heat_3d(1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1);

    let grid_bound = args.grid_bounds();

    // Create domains
    let mut buffer_1 = OwnedDomain::new(grid_bound);
    let mut buffer_2 = OwnedDomain::new(grid_bound);
    let mut input_domain = buffer_1.as_slice_domain();
    let mut output_domain = buffer_2.as_slice_domain();

    // Create BC
    let bc = ConstantCheck::new(1.0, grid_bound);

    let direct_solver = DirectFrustrumSolver {
        bc: &bc,
        stencil: &stencil,
        stencil_slopes: stencil.slopes(),
        chunk_size: args.chunk_size,
    };

    // Create AP Solver
    let solver_parameters = args.solver_parameters();
    let mut solver =
        generate_ap_solver(&stencil, direct_solver, &solver_parameters);

    solver.print_report();
    if args.write_dot {
        solver.to_dot_file(&args.dot_path());
    }
    if args.gen_only {
        args.finish();
        std::process::exit(0);
    }

    if args.write_images {
        write_vtk3d(&input_domain, &args.frame_name(0));
    }

    let mut global_time = 0;
    for t in 1..args.images {
        let now = Instant::now();
        solver.apply(&mut input_domain, &mut output_domain, global_time);
        let elapsed_time = now.elapsed();
        eprintln!("{}", elapsed_time.as_nanos() as f64 / 1000000000.0);

        global_time += args.steps_per_image;
        std::mem::swap(&mut input_domain, &mut output_domain);
        if args.write_images {
            write_vtk3d(&input_domain, &args.frame_name(t));
        }
    }

    args.finish();
}
