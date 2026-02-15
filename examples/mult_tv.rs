use nhls::domain::*;
use nhls::image::*;
use nhls::util::*;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Domain {
    S1,
    S2,
}

fn main() {
    let n = 400;
    let n_steps = 400;
    let lines = n_steps;
    let mut input = vec![0.0; n];
    let mut output = vec![0.0; n];
    let mut input_mask = vec![Domain::S1; n];
    let mut output_mask = vec![Domain::S2; n];

    let variance = 24.0;
    let n_f = n as f64;
    let sigma_sq: f64 = (n_f / variance) * (n_f / variance);
    let mut input_boundaries = Vec::new();
    let mut output_boundaries = Vec::new();
    for i in 0..n {
        let x = (i as f64) - (n_f / 2.0);
        let exp = -x * x / (2.0 * sigma_sq);
        let v = exp.exp();
        input[i] = v;
        if v > 0.5 {
            input_mask[i] = Domain::S1;
        } else {
            input_mask[i] = Domain::S2;
        }

        if i > 0 && input_mask[i - 1] != input_mask[i] {
            input_boundaries.push(i);
        }
    }

    // s1 adds energy
    // s2 removes energy

    let s1 = |args: [f64; 3]| {
        let l = args[0];
        let c = args[1];
        let r = args[2];
        0.4 * l + 0.4 * c + 0.4 * r
    };

    let s2 = |args: [f64; 3]| {
        let l = args[0];
        let c = args[1];
        let r = args[2];
        0.2 * l + 0.2 * c + 0.2 * r
    };

    let oracle = |domains: [Domain; 3], args: [f64; 3]| {
        //let l = args[0];
        let c = args[1];
        //let r = args[2];
        match domains {
            // No Change
            [Domain::S1, Domain::S1, Domain::S1] => (0, Domain::S1),
            [Domain::S2, Domain::S2, Domain::S2] => (0, Domain::S2),
            [Domain::S1, Domain::S1, Domain::S2] => {
                if c > 0.5 {
                   (-1 , Domain::S2)
                } else {
                   (0 , Domain::S1)
                }
            },
            [Domain::S2, Domain::S1, Domain::S1] => {
                if c > 0.5 {
                   (1, Domain::S2) 
                } else {
                   (0, Domain::S1)
                }
            }
            [Domain::S2, Domain::S1, Domain::S2] => {
                println!("Shouldn't hapen");
                (0, Domain::S1)
            }
            [Domain::S2, Domain::S2, Domain::S1] => {
                if c > 0.5 {
                   (0, Domain::S2)
                } else {
                    (-1, Domain::S1)
                }
            },
            [Domain::S1, Domain::S2, Domain::S2] => {
                if c > 0.5 {
                    (0, Domain::S2)
                } else {
                   (1, Domain::S1)
                }
            },
            [Domain::S1, Domain::S2, Domain::S1] => {
                println!("Shouldn't hapen");
                (0, Domain::S2)
            }
        }
    };

    let bounds = AABB::new(matrix![0, (n as i32) - 1]);
    let mut data_img = Image1D::new(bounds, lines);
    let mut mask_img = Image1D::new(bounds, lines);

    for i in 0..n {
        match input_mask[i] {
            Domain::S1 => {
                output[i] = 0.0;
            }
            Domain::S2 => {
                output[i] = 1.0;
            }
        }
    }
    data_img.add_line(0, &input);
    mask_img.add_line(0, &output);
    for t in 1..n_steps {
        //output_mask = input_mask;
        println!("i: {}, o: {}", input_boundaries.len(), output_boundaries.len());
        for i in 0..n {
            let l_i = if i == 0 { n - 1 } else { i - 1 };

            let r_i = if i == n - 1 { 0 } else { i + 1 };

            let args_i = [input[l_i], input[i], input[r_i]];

            match input_mask[i] {
                Domain::S1 => {
                    output[i] = s1(args_i);
                }
                Domain::S2 => {
                    output[i] = s2(args_i);
                }
            }

            let args_o = [output[l_i], output[i], output[r_i]];
            if input_boundaries.contains(&i) {
                let domains = [input_mask[l_i], input_mask[i], input_mask[r_i]];
                let (ix, d) = oracle(domains, args_o);
                output_mask[i] = d;
                output_boundaries.push(((i as i32) + ix) as usize);
            }
            //output_mask[i] = oracle(domains, args_o);
        }

        // Update images
        std::mem::swap(&mut input, &mut output);
        std::mem::swap(&mut input_mask, &mut output_mask);
        std::mem::swap(&mut input_boundaries, &mut output_boundaries);
        output_boundaries.clear();

        for i in 0..n {
            match input_mask[i] {
                Domain::S1 => {
                    output[i] = 0.0;
                }
                Domain::S2 => {
                    output[i] = 1.0;
                }
            }
        }
        data_img.add_line(t, &input);
        mask_img.add_line(t, &output);
    }

    data_img.write(&"data_img.png");
    mask_img.write(&"mask_img.png");
}
