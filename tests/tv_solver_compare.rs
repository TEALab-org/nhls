use float_cmp::assert_approx_eq;
use nhls::domain::*;
use nhls::fft_solver::*;
use nhls::init::*;
use nhls::solver::*;
use nhls::standard_stencils::*;
use nhls::time_varying::*;
use nhls::util::*;

#[test]
fn tv_rotating_advection_compare() {
    // Grid size
    let grid_bound = AABB::new(matrix![333, 394; 5, 66]);

    let n_steps = 400;

    let threads = 8;

    let chunk_size = 100;

    let plan_type = PlanType::Estimate;

    let stencil = RotatingAdvectionStencil::new(100.0, 0.2);

    // Create domains
    let buffer_size = grid_bound.buffer_size();
    let mut direct_domain_1 = OwnedDomain::new(grid_bound);
    let mut direct_domain_2 = OwnedDomain::new(grid_bound);
    let mut direct_input_domain = direct_domain_1.as_slice_domain();
    let mut direct_output_domain = direct_domain_2.as_slice_domain();
    let mut fft_buffer_1 = OwnedDomain::new(grid_bound);
    let mut fft_buffer_2 = OwnedDomain::new(grid_bound);
    let mut fft_input_domain = fft_buffer_1.as_slice_domain();
    let mut fft_output_domain = fft_buffer_2.as_slice_domain();

    // Fill in with IC values (use normal dist for spike in the middle)
    normal_ic_2d(&mut direct_input_domain, chunk_size);
    normal_ic_2d(&mut fft_input_domain, chunk_size);

    let direct_solver = AP2DDirectSolver::new(&stencil);
    direct_solver.apply(
        &mut direct_input_domain,
        &mut direct_output_domain,
        &Bounds::zeros(),
        n_steps,
        0,
        threads,
    );

    let planner_params = PlannerParameters {
        plan_type,
        cutoff: 20,
        ratio: 0.5,
        chunk_size,
    };
    let mut solver = TVAPSolver::new(
        &stencil,
        grid_bound,
        n_steps,
        threads,
        &planner_params,
        direct_solver,
    );
    solver.apply(&mut fft_input_domain, &mut fft_output_domain, 0);

    for i in 0..buffer_size {
        assert_approx_eq!(
            f64,
            fft_output_domain.buffer()[i],
            direct_output_domain.buffer()[i],
            epsilon = 0.0000000000001
        );
    }
}
