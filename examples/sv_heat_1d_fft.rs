use core::f64;
use nhls::domain::*;
use nhls::image_1d_example::*;
use nhls::initial_conditions::*;
use nhls::mirror_domain::*;
use std::time::*;

fn main() {
    let args = Args::cli_setup("sv_heat_1d_fft");

    // Grid size
    let mut grid_bound = args.grid_bounds();
    grid_bound.bounds[(0, 1)] = (args.domain_size / 2) as i32 - 1;
    println!("Grid bound: {}", grid_bound);

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.2);

    // Create domains
    let mut buffer_11 = OwnedDomain::new(grid_bound);
    let mut buffer_12 = OwnedDomain::new(grid_bound);
    let mut input_domain_1 = buffer_11.as_slice_domain();
    let mut output_domain_1 = buffer_12.as_slice_domain();

    let mut buffer_21 = OwnedDomain::new(grid_bound);
    let mut buffer_22 = OwnedDomain::new(grid_bound);
    let mut input_domain_2 = buffer_21.as_slice_domain();
    let mut output_domain_2 = buffer_22.as_slice_domain();

    // Setup initial conditions
    let ic_type = args.ic_type.to_ic_type(args.ic_dial);
    generate_ic_1d(&mut input_domain_1, ic_type, args.chunk_size);
    generate_ic_1d(&mut input_domain_2, ic_type, args.chunk_size);

    let direct_solver = SV1DDirectSolver::new(&stencil);
    let solver_params = args.solver_parameters();
    let solver = SVSolver::new(&stencil, &solver_params, direct_solver);

    if args.gen_only {
        args.finish();
        std::process::exit(0);
    }

    // Apply direct solver
    let mut global_time = 0;
    for _ in 1..args.lines {
        let now = Instant::now();
        solver.apply(
            &mut input_domain_1,
            &mut output_domain_1,
            &mut input_domain_2,
            &mut output_domain_2,
            global_time,
        );
        let elapsed_time = now.elapsed();
        eprintln!("{}", elapsed_time.as_nanos() as f64 / 1000000000.0);

        global_time += args.steps_per_line;
        std::mem::swap(&mut input_domain_1, &mut output_domain_1);
        std::mem::swap(&mut input_domain_2, &mut output_domain_2);
    }

    args.finish();
}
