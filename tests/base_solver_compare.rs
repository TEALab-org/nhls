use float_cmp::assert_approx_eq;
use nhls::domain::*;
use nhls::fft_solver::*;
use nhls::initial_conditions::normal_impulse::*;
use nhls::solver::*;
use nhls::stencil::*;
use nhls::util::*;

#[test]
fn heat_1d_p_compare() {
    // Grid size
    let grid_bound = AABB::new(matrix![0, 999]);

    let n_steps = 400;

    let chunk_size = 100;

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.5);

    // Create domains
    let buffer_size = grid_bound.buffer_size();
    let mut direct_input_domain = OwnedDomain::new(grid_bound);
    let mut direct_output_domain = OwnedDomain::new(grid_bound);

    let mut fft_input_domain = OwnedDomain::new(grid_bound);
    let mut fft_output_domain = OwnedDomain::new(grid_bound);

    // Fill in with IC values (use normal dist for spike in the middle)
    normal_ic_1d(&mut direct_input_domain, 25.0, chunk_size);
    normal_ic_1d(&mut fft_input_domain, 25.0, chunk_size);

    let plan_type = PlanType::Estimate;
    let mut periodic_solver = PeriodicSolver::create(
        &stencil,
        fft_output_domain.buffer_mut(),
        &grid_bound,
        n_steps,
        plan_type,
        chunk_size,
        8,
    );
    periodic_solver.apply(&mut fft_input_domain, &mut fft_output_domain);

    direct_periodic_apply(
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
fn heat_2d_p_compare() {
    // Grid size
    let grid_bound = AABB::new(matrix![2, 81; 14, 87]);

    let n_steps = 400;

    let chunk_size = 100;

    let stencil = nhls::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);

    // Create domains
    let buffer_size = grid_bound.buffer_size();
    let mut direct_input_domain = OwnedDomain::new(grid_bound);
    let mut direct_output_domain = OwnedDomain::new(grid_bound);

    let mut fft_input_domain = OwnedDomain::new(grid_bound);
    let mut fft_output_domain = OwnedDomain::new(grid_bound);

    normal_ic_2d(&mut direct_input_domain, 25.0, chunk_size);
    normal_ic_2d(&mut fft_input_domain, 25.0, chunk_size);

    let plan_type = PlanType::Estimate;
    let mut periodic_solver = PeriodicSolver::create(
        &stencil,
        fft_output_domain.buffer_mut(),
        &grid_bound,
        n_steps,
        plan_type,
        chunk_size,
        8,
    );
    periodic_solver.apply(&mut fft_input_domain, &mut fft_output_domain);

    direct_periodic_apply(
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
fn periodic_compare() {
    {
        let steps = 1;
        let chunk_size = 3;
        let bound = AABB::new(matrix![0, 99]);
        let stencil = Stencil::new(
            [[-1], [-2], [0], [3], [4], [1]],
            |args: &[f64; 6]| {
                let c = 1.0 / 6.0;
                args.iter().map(|x| c * x).sum()
            },
        );

        let n_r = bound.buffer_size();
        let mut input_a = AlignedVec::new(n_r);
        for i in 0..n_r {
            input_a[i] = i as f64;
        }

        let mut domain_a_input = OwnedDomain::new(bound);
        let mut domain_b_input = OwnedDomain::new(bound);
        let mut domain_a_output = OwnedDomain::new(bound);
        let mut domain_b_output = OwnedDomain::new(bound);

        domain_a_input.par_set_values(|coord| coord[0] as f64, chunk_size);
        domain_b_input.par_set_values(|coord| coord[0] as f64, chunk_size);

        direct_periodic_apply(
            &stencil,
            &mut domain_a_input,
            &mut domain_a_output,
            steps,
            chunk_size,
        );
        let plan_type = PlanType::Estimate;
        let mut periodic_solver = PeriodicSolver::create(
            &stencil,
            domain_b_output.buffer_mut(),
            &bound,
            steps,
            plan_type,
            chunk_size,
            8,
        );
        periodic_solver.apply(&mut domain_b_input, &mut domain_b_output);

        for i in 0..n_r {
            assert_approx_eq!(
                f64,
                domain_a_output.buffer()[i],
                domain_b_output.buffer()[i],
                epsilon = 0.0000000000001
            );
        }
    }
}
