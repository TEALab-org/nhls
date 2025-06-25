use float_cmp::assert_approx_eq;
use nhls::direct_solver::*;
use nhls::domain::*;
use nhls::initial_conditions::rand::*;
use nhls::util::*;
use nhls::SolverInterface;

#[test]
fn direct_opt_3pt1d_compare() {
    // Params
    let grid_bound = AABB::new(matrix![0, 99]);
    let n_steps = 40;
    let chunk_size = 10;
    let threads = 8;
    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.25);
    // 0.25, 0.25, 0.5 weights

    // Create buffers / domains
    let mut opt_buffer_1 = OwnedDomain::new(grid_bound);
    let mut opt_buffer_2 = OwnedDomain::new(grid_bound);
    let mut naive_buffer_1 = OwnedDomain::new(grid_bound);
    let mut naive_buffer_2 = OwnedDomain::new(grid_bound);
    let mut opt_in = opt_buffer_1.as_slice_domain();
    let mut opt_out = opt_buffer_2.as_slice_domain();
    let mut naive_in = naive_buffer_1.as_slice_domain();
    let mut naive_out = naive_buffer_2.as_slice_domain();

    // Setup ICs
    rand_ic(&mut opt_in, 1024, chunk_size);
    naive_in.buffer_mut().copy_from_slice(opt_in.buffer());

    // Opt
    let mut opt_solver =
        Direct3Pt1DSolver::new(&stencil, n_steps, threads, chunk_size);
    opt_solver.apply(&mut opt_in, &mut opt_out, 0);

    // Naive
    let bc = ConstantCheck::new(0.0, grid_bound);
    box_apply(
        &bc,
        &stencil,
        &mut naive_in,
        &mut naive_out,
        n_steps,
        0,
        chunk_size,
    );

    // Compare
    let buffer_size = grid_bound.buffer_size();
    for i in 0..buffer_size {
        assert_approx_eq!(
            f64,
            opt_out.buffer()[i],
            naive_out.buffer()[i],
            epsilon = 0.000000000000000001
        );
    }
}

#[test]
fn direct_opt_5pt2d_compare() {
    // Params
    //let grid_bound = AABB::new(matrix![0, 19; 0, 19]);
    let grid_bound = AABB::new(matrix![0, 13; 0, 18]);
    let n_steps = 13;
    let chunk_size = 100;
    let threads = 1;
    let stencil = nhls::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);

    // Create buffers / domains
    let mut opt_buffer_1 = OwnedDomain::new(grid_bound);
    let mut opt_buffer_2 = OwnedDomain::new(grid_bound);
    let mut naive_buffer_1 = OwnedDomain::new(grid_bound);
    let mut naive_buffer_2 = OwnedDomain::new(grid_bound);
    let mut opt_in = opt_buffer_1.as_slice_domain();
    let mut opt_out = opt_buffer_2.as_slice_domain();
    let mut naive_in = naive_buffer_1.as_slice_domain();
    let mut naive_out = naive_buffer_2.as_slice_domain();

    // Setup ICs
    rand_ic(&mut opt_in, 1024, chunk_size);
    naive_in.buffer_mut().copy_from_slice(opt_in.buffer());

    // Opt
    let mut opt_solver =
        Direct5Pt2DSolver::new(&stencil, n_steps, threads, chunk_size);
    opt_solver.apply(&mut opt_in, &mut opt_out, 0);

    // Naive
    let bc = ConstantCheck::new(0.0, grid_bound);
    box_apply(
        &bc,
        &stencil,
        &mut naive_in,
        &mut naive_out,
        n_steps,
        0,
        chunk_size,
    );

    for i in 0..grid_bound.buffer_size() {
        opt_in.buffer_mut()[i] = opt_out.buffer()[i] - naive_out.buffer()[i];
    }

    // Compare
    let buffer_size = grid_bound.buffer_size();
    for i in 0..buffer_size {
        assert_approx_eq!(
            f64,
            opt_out.buffer()[i],
            naive_out.buffer()[i],
            epsilon = 0.000000000000000001
        );
    }
}
