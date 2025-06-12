use nhls::ap_solver::generate_solver::*;
use nhls::ap_solver::planner::PlannerParameters;
use nhls::ap_solver::solver::SolverInterface;
use nhls::domain::*;
use nhls::fft_solver::DirectFrustrumSolver;
use nhls::image_1d_example::*;

fn main() {
    let (args, output_image_path) = Args::cli_parse("heat_1d_ap_fft");

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.5);
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
    let planner_params = PlannerParameters {
        plan_type: args.plan_type,
        cutoff: args.cutoff,
        ratio: args.ratio,
        chunk_size: args.chunk_size,
        threads: args.threads,
        steps: args.steps_per_line,
        aabb: grid_bound,
    };
    let mut solver =
        generate_ap_solver_1d(&stencil, direct_solver, &planner_params);

    solver.print_report();

    if args.write_dot {
        solver.to_dot_file(&args.dot_path());
    }

    if args.gen_only {
        args.save_wisdom();
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
        solver.apply(&mut input_domain, &mut output_domain, global_time);
        global_time += args.steps_per_line;
        std::mem::swap(&mut input_domain, &mut output_domain);
        if let Some(i) = img.as_mut() {
            i.add_line(t, input_domain.buffer());
        }
    }

    if let Some(i) = img {
        i.write(&output_image_path);
    }

    args.save_wisdom();
}
