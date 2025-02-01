use float_cmp::assert_approx_eq;
use nhls::domain::*;
use nhls::fft_solver::*;
use nhls::init::*;
use nhls::solver::*;
use nhls::util::*;

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
    normal_ic_1d(&mut direct_input_domain, chunk_size);
    normal_ic_1d(&mut fft_input_domain, chunk_size);

    // Create BC
    let bc = ConstantCheck::new(1.0, grid_bound);

    // Create AP Solver
    let cutoff = 40;
    let ratio = 0.5;
    let plan_type = PlanType::Estimate;
    let fft_solver = APSolver::new(
        &bc, &stencil, grid_bound, n_steps, plan_type, cutoff, ratio,
        chunk_size,
    );
    fft_solver.apply(&mut fft_input_domain, &mut fft_output_domain);

    box_apply(
        &bc,
        &stencil,
        &mut direct_input_domain,
        &mut direct_output_domain,
        n_steps,
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
    normal_ic_2d(&mut direct_input_domain, chunk_size);
    normal_ic_2d(&mut fft_input_domain, chunk_size);

    // Create BC
    let bc = ConstantCheck::new(1.0, grid_bound);

    // Create AP Solver
    let cutoff = 40;
    let ratio = 0.5;
    let plan_type = PlanType::Estimate;
    let fft_solver = APSolver::new(
        &bc, &stencil, grid_bound, n_steps, plan_type, cutoff, ratio,
        chunk_size,
    );
    fft_solver.apply(&mut fft_input_domain, &mut fft_output_domain);

    box_apply(
        &bc,
        &stencil,
        &mut direct_input_domain,
        &mut direct_output_domain,
        n_steps,
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
