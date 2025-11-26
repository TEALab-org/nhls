use nhls::domain::*;
use nhls::image::*;
use nhls::initial_conditions::normal_impulse::normal_ic_2d;
use nhls::util::*;
use std::path::PathBuf;

const OFFSETS: [[i32; 2]; 4] = [[1, 0], [0, -1], [-1, 0], [0, 1]];

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Domain {
    S1, // Remove energy 0.2
    S2, // Conserve Energy 0.3
    S3, // Add Energy 0.4
}

fn mask_to_domain(
    mask: &OwnedDomainMask<2, Domain>,
    output: &mut OwnedDomain<2>,
    chunk_size: usize,
) {
    let value_f = |coord| match mask.view(&coord) {
        Domain::S1 => 0.0,
        Domain::S2 => 0.5,
        Domain::S3 => 1.0,
    };
    output.par_set_values(value_f, chunk_size);
}

fn domain_frame_name(i: usize) -> String {
    format!("target/mult_tv_2d_01/domain_frame_{i:04}.png")
}

fn mask_frame_name(i: usize) -> String {
    format!("target/mult_tv_2d_01/mask_frame_{i:04}.png")
}

fn main() {
    let n = 400;
    let variance = 10.0;
    let bounds = AABB::new(matrix![0, n - 1; 0, n - 1]);
    let n_steps = 100;
    let chunk_size = 10000;

    let offsets: [Coord<2>; 4] =
        std::array::from_fn(|i| Coord::from_column_slice(&OFFSETS[i]));

    let mut input = OwnedDomain::new(bounds);
    let mut output = OwnedDomain::new(bounds);
    let mut mask_image_domain = OwnedDomain::new(bounds);

    let mut input_mask = OwnedDomainMask::new(bounds, Domain::S1);
    let mut output_mask = OwnedDomainMask::new(bounds, Domain::S1);

    normal_ic_2d(&mut input, variance, chunk_size);

    let s1w = 0.9 / 5.0;
    let s1 = |args: [f64; 5]| {
        args.iter().map(|a| a * s1w).sum::<f64>()
    };

    let s2w = 1.0 / 5.0;
    let s2 = |args: [f64; 5]| {
        args.iter().map(|a| a * s2w).sum::<f64>()
    };

    let s3w = 1.1 / 5.0;
    let s3 = |args: [f64; 5]| {
        //0.1 * args[0] + 0.4 * args[1] + 0.1 * args[2] + 0.1 * args[3] + 0.4 * args[0]
        args.iter().map(|a| a * s3w).sum::<f64>()
    };

    let oracle = |value: f64| match value {
        v if v > 0.7 => Domain::S1,
        v if v >= 0.3 && v <= 0.7 => Domain::S2,
        v if v < 0.3 => Domain::S3,
        _ => {
            panic!("Unknown ORACLE state");
        }
    };

    // Set input mask
    for coord in bounds.coord_iter() {
        let v: f64 = input.view(&coord);
        let d = oracle(v);
        input_mask.set_coord(&coord, d);
    }

    // output input / mask image
    mask_to_domain(&input_mask, &mut mask_image_domain, chunk_size);
    image2d(&input, &domain_frame_name(0));
    image2d(&mask_image_domain, &mask_frame_name(0));

    for t in 1..n_steps {
        // Run Stencils, update output
        for coord in bounds.coord_iter() {
          let mut args = [0.0, 0.0, 0.0, 0.0, 0.0];
          args[4] = input.view(&coord);
          for (i, offset) in offsets.iter().enumerate() {
            args[i] = input.view(&bounds.periodic_coord(&(coord + offset)));
          }
          output.set_coord(&coord, match input_mask.view(&coord) {
            Domain::S1 => s1(args),
            Domain::S2 => s2(args),
            Domain::S3 => s3(args),
          });
        }

        // Run oracle, update output mask
        for coord in bounds.coord_iter() {
            // Is a boundary cell?
            let d = input_mask.view(&coord);
            let is_not_boundary = offsets.iter().map(|o| {
                let n_c = bounds.periodic_coord(&(coord + o));
                input_mask.view(&n_c)
            }).all(|nd| nd == d);

            if !is_not_boundary {
                let d = oracle(output.view(&coord));
                output_mask.set_coord(&coord, d);
            } else {
                let d = input_mask.view(&coord);
                output_mask.set_coord(&coord, d);
            }
        }

        // Save images
        mask_to_domain(&output_mask, &mut mask_image_domain, chunk_size);
        image2d(&output, &domain_frame_name(t));
        image2d(&mask_image_domain, &mask_frame_name(t));

        // Swap buffers
        std::mem::swap(&mut input, &mut output);
        std::mem::swap(&mut input_mask, &mut output_mask);
    }
}
