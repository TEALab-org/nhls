use float_cmp::assert_approx_eq;
use nhls::ap_solver::*;
use nhls::direct_solver::*;
use nhls::domain::*;
use nhls::initial_conditions::normal_impulse::*;
use nhls::util::*;
use nhls::SolverInterface;

pub const TEST_SOLVE_THREADS: usize = 8;

#[test]
fn heat_1d_ap_compare() {
    // Grid size
    let grid_bound = AABB::new(matrix![0, 999]);

    let n_steps = 400;

    let chunk_size = 100;

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.5);

    // Create domains
    let buffer_size = grid_bound.buffer_size();
    let mut direct_input_domain = OwnedDomain::new(grid_bound);
    let mut direct_output_domain = OwnedDomain::new(grid_bound);
    let mut fft_buffer_1 = OwnedDomain::new(grid_bound);
    let mut fft_buffer_2 = OwnedDomain::new(grid_bound);
    let mut fft_input_domain = fft_buffer_1.as_slice_domain();
    let mut fft_output_domain = fft_buffer_2.as_slice_domain();

    // Fill in with IC values (use normal dist for spike in the middle)
    normal_ic_1d(&mut direct_input_domain, 25.0, chunk_size);
    normal_ic_1d(&mut fft_input_domain, 25.0, chunk_size);

    // Create BC
    let bc = ConstantCheck::new(1.0, grid_bound);

    // Create AP Solver
    let solver_params = SolverParameters {
        plan_type: PlanType::Estimate,
        cutoff: 40,
        chunk_size,
        threads: TEST_SOLVE_THREADS,
        aabb: grid_bound,
        steps: n_steps,
        ..Default::default()
    };
    let direct_solver = DirectFrustrumSolver {
        bc: &bc,
        stencil: &stencil,
        stencil_slopes: stencil.slopes(),
        chunk_size,
    };
    let mut fft_solver =
        generate_ap_solver_1d(&stencil, direct_solver, &solver_params);
    fft_solver.apply(&mut fft_input_domain, &mut fft_output_domain, 0);

    box_apply(
        &bc,
        &stencil,
        &mut direct_input_domain,
        &mut direct_output_domain,
        n_steps,
        0,
        chunk_size,
    );

    for i in 0..buffer_size {
        assert_approx_eq!(
            f64,
            fft_output_domain.buffer()[i],
            direct_output_domain.buffer()[i],
            epsilon = 0.0000000000001
        );
    }
}

#[test]
fn heat_2d_ap_compare() {
    // Grid size
    let grid_bound = AABB::new(matrix![333, 391; 5, 61]);

    let n_steps = 400;

    let chunk_size = 100;

    let stencil = nhls::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);

    // Create domains
    let buffer_size = grid_bound.buffer_size();
    let mut direct_input_domain = OwnedDomain::new(grid_bound);
    let mut direct_output_domain = OwnedDomain::new(grid_bound);
    let mut fft_buffer_1 = OwnedDomain::new(grid_bound);
    let mut fft_buffer_2 = OwnedDomain::new(grid_bound);
    let mut fft_input_domain = fft_buffer_1.as_slice_domain();
    let mut fft_output_domain = fft_buffer_2.as_slice_domain();

    // Fill in with IC values (use normal dist for spike in the middle)
    normal_ic_2d(&mut direct_input_domain, 25.0, chunk_size);
    normal_ic_2d(&mut fft_input_domain, 25.0, chunk_size);

    // Create BC
    let bc = ConstantCheck::new(1.0, grid_bound);

    // Create AP Solver
    let solver_params = SolverParameters {
        cutoff: 40,
        chunk_size,
        threads: TEST_SOLVE_THREADS,
        aabb: grid_bound,
        steps: n_steps,
        ..Default::default()
    };
    let direct_solver = DirectFrustrumSolver {
        bc: &bc,
        stencil: &stencil,
        stencil_slopes: stencil.slopes(),
        chunk_size,
    };

    let mut fft_solver =
        generate_ap_solver_2d(&stencil, direct_solver, &solver_params);
    fft_solver.apply(&mut fft_input_domain, &mut fft_output_domain, 0);

    box_apply(
        &bc,
        &stencil,
        &mut direct_input_domain,
        &mut direct_output_domain,
        n_steps,
        0,
        chunk_size,
    );

    for i in 0..buffer_size {
        assert_approx_eq!(
            f64,
            fft_output_domain.buffer()[i],
            direct_output_domain.buffer()[i],
            epsilon = 0.000001
        );
    }
}
