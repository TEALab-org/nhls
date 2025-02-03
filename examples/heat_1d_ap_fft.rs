use nhls::domain::*;
use nhls::fft_solver::*;
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

    // Create AP Solver
    let planner_params = PlannerParameters {
        plan_type: args.plan_type,
        cutoff: 40,
        ratio: 0.5,
        chunk_size: args.chunk_size,
    };
    let solver = APSolver::new(
        &bc,
        &stencil,
        grid_bound,
        args.steps_per_line,
        &planner_params,
    );
    if args.write_dot {
        println!("WRITING DOT FILE");
        let mut dot_path = args.output_dir.clone();
        dot_path.push("plan.dot");
        solver.to_dot_file(&dot_path);
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
