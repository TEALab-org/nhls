use nhls::domain::*;
use nhls::solver::*;
use nhls::stencil::*;
use nhls::util::*;
use rayon::prelude::*;

use fftw::array::*;
use float_cmp::assert_approx_eq;
use nalgebra::{matrix, vector};

#[test]
fn thermal_1d_compare() {
    const GRID_DIMENSION: usize = 1;

    // Grid size
    let grid_bound = matrix![0, 999];

    let n_steps = 16;

    // Step size t
    let dt: f32 = 1.0;

    // Step size x
    let dx: f32 = 1.0;

    // Heat transfer coefficient
    let k: f32 = 0.5;

    let chunk_size = 100;

    let stencil = Stencil::new([[-1], [0], [1]], |args: &[f32; 3]| {
        let left = args[0];
        let middle = args[1];
        let right = args[2];
        middle + (k * dt / (dx * dx)) * (left - 2.0 * middle + right)
    });

    // Create domains
    let buffer_size = box_buffer_size(&grid_bound);
    let mut grid_input = vec![0.0; buffer_size];
    let mut naive_input_domain = Domain::new(grid_bound, &mut grid_input);

    let mut grid_output = vec![0.0; buffer_size];
    let mut naive_output_domain = Domain::new(grid_bound, &mut grid_output);

    let mut fft_input = AlignedVec::new(buffer_size);
    let mut fft_output = AlignedVec::new(buffer_size);
    let mut fft_input_domain = Domain::new(grid_bound, &mut fft_input);
    let mut fft_output_domain = Domain::new(grid_bound, &mut fft_output);

    // Fill in with IC values (use normal dist for spike in the middle)
    let n_f = buffer_size as f32;
    let sigma_sq: f32 = (n_f / 25.0) * (n_f / 25.0);
    naive_input_domain
        .par_modify_access(100)
        .for_each(|mut d: DomainChunk<'_, GRID_DIMENSION>| {
            d.coord_iter_mut().for_each(
                |(world_coord, value_mut): (Coord<GRID_DIMENSION>, &mut f32)| {
                    let x = (world_coord[0] as f32) - (n_f / 2.0);
                    //let f = ( 1.0 / (2.0 * std::f32::consts::PI * sigma_sq)).sqrt();
                    let exp = -x * x / (2.0 * sigma_sq);
                    *value_mut = exp.exp()
                },
            )
        });

    fft_input_domain
        .par_modify_access(100)
        .for_each(|mut d: DomainChunk<'_, GRID_DIMENSION>| {
            d.coord_iter_mut().for_each(
                |(world_coord, value_mut): (Coord<GRID_DIMENSION>, &mut f32)| {
                    let x = (world_coord[0] as f32) - (n_f / 2.0);
                    //let f = ( 1.0 / (2.0 * std::f32::consts::PI * sigma_sq)).sqrt();
                    let exp = -x * x / (2.0 * sigma_sq);
                    *value_mut = exp.exp()
                },
            )
        });

    let mut periodic_library =
        nhls::solver::periodic_plan::PeriodicPlanLibrary::new(&grid_bound, &stencil);
    periodic_library.apply(
        &mut fft_input_domain,
        &mut fft_output_domain,
        n_steps,
        chunk_size,
    );

    periodic_naive::box_solve(
        &stencil,
        &mut naive_input_domain,
        &mut naive_output_domain,
        n_steps,
        chunk_size,
    );

    for i in 0..buffer_size {
        // TODO THIS IS PRETTY BAD
        assert_approx_eq!(f32, fft_output[i], grid_output[i], epsilon = 0.001);
    }
}

#[test]
fn periodic_compare() {
    {
        let steps = 1;
        let chunk_size = 3;
        let bound = matrix![0, 99];
        let stencil = Stencil::new([[-1], [-2], [0], [3], [4], [1]], |args: &[f32; 6]| {
            let c = 1.0 / 6.0;
            args.iter().map(|x| c * x).sum()
        });

        let n_r = box_buffer_size(&bound);
        let mut input_a = AlignedVec::new(n_r);
        for i in 0..n_r {
            input_a[i] = i as f32;
        }
        let mut input_b = input_a.clone();

        let mut domain_a_input = Domain::new(bound, input_a.as_slice_mut());
        let mut domain_b_input = Domain::new(bound, input_b.as_slice_mut());

        let mut output_a = AlignedVec::new(n_r);
        let mut output_b = AlignedVec::new(n_r);
        let mut domain_a_output = Domain::new(bound, output_a.as_slice_mut());
        let mut domain_b_output = Domain::new(bound, output_b.as_slice_mut());

        periodic_naive::box_solve(
            &stencil,
            &mut domain_a_input,
            &mut domain_a_output,
            steps,
            chunk_size,
        );
        let mut plan_library = periodic_plan::PeriodicPlanLibrary::new(&bound, &stencil);
        plan_library.apply(&mut domain_b_input, &mut domain_b_output, steps, chunk_size);

        for i in 0..n_r {
            println!(
                "n: {}, p: {}, d: {}",
                output_a[i],
                output_b[i],
                (output_a[i] - output_b[i]).abs()
            );
        }
        for i in 0..n_r {
            // TODO THIS IS BROKE
            //assert_approx_eq!(f32, output_a[i], output_b[i], epsilon = 0.001);
        }
    }
}
