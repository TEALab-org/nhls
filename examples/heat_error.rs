use nhls::csv::*;
use nhls::fft_solver::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::image_init::*;
use nhls::init::*;
use nhls::solver::*;
use nhls::stencil::*;
use nhls::time_varying::*;
use std::time::*;

fn main() {
    let n_threads = 8;
    let chunk_size = 1000;
    let T = 60;
    let grid_bound = AABB::new(matrix![0, 29; 0, 29]);
    let stencil = Stencil::new(
        [[1, 0], [0, -1], [-1, 0], [0, 1], [0, 0]],
        move |args: &[f64; 5]| {
            0.2 * args.iter().sum::<f64>() 
        },
    );
    // Create BC
    let bc = ConstantCheck::new(0.0, grid_bound);

    // Create AP Solver
    let planner_params = PlannerParameters {
        plan_type: PlanType::Estimate,
        cutoff: 10,
        ratio: 0.5,
        chunk_size,
    };
    let solver =
        APSolver::new(&bc, &stencil, grid_bound, T, &planner_params, n_threads);
    solver.print_report();

    // Create domains
    let mut buffer_1 = OwnedDomain::new(grid_bound);
    let mut buffer_2 = OwnedDomain::new(grid_bound);
    let mut input_domain = buffer_1.as_slice_domain();
    let mut output_domain = buffer_2.as_slice_domain();

    for ix in 0..10 {
        for iy in 0..10 {
           input_domain.set_coord(&vector![ix + 10, iy + 10], 1.0); 
        }
    }

    write_csv_2d(&input_domain, &"/Users/russell/projects/stencil_research/error_plots/t0.csv");

    // TODO: init values
    // write csv

    let mut global_time = 0;
    solver.apply(&mut input_domain, &mut output_domain, global_time);
    global_time += T;
    std::mem::swap(&mut input_domain, &mut output_domain);
    write_csv_2d(&input_domain, &"/Users/russell/projects/stencil_research/error_plots/t1.csv");

    solver.apply(&mut input_domain, &mut output_domain, global_time);
    global_time += T;
    std::mem::swap(&mut input_domain, &mut output_domain);
    write_csv_2d(&input_domain, &"/Users/russell/projects/stencil_research/error_plots/t2.csv");
}
