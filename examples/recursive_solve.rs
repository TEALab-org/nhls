// Recursive domain decomposition.
//
// We need boundary condition functions
//
// What is value out of bounds

//
// Face vectors

#[derive(Debug, Copy, Clone)]
pub enum Boundary {
    SPACE,
    PERIODIC,
    FIXED,
}

#[derive(Debug)]
pub struct FFTParams {
    sigma: usize,
    cutoff: usize,
    ratio: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct FFTSolveArgs {
    boundaries: [Boundary; 2],
    t0: usize,
    t_end: usize,
    x0: usize,
    x_end: usize,
}

pub fn space(i: usize) -> String {
    let mut result = String::new();
    for _ in 0..i {
        result += "  ";
    }
    result
}

pub fn recursive_solve(args: FFTSolveArgs, p: FFTParams, level: usize) {
    let sp = space(level);
    println!("{}RS: l: {}, args: {:?}, p: {:?}", sp, level, args, p);

    // Are we below cutoff?
    let x_size = args.x_end - args.x0;
    if x_size < p.cutoff {
        println!("{}* Recursive solve to {}", sp, args.t_end);
        return;
    }

    // Compute FFT Solve
    let solve_size = (p.ratio * x_size as f32) as usize;

    // Check rounding here
    let delta_t = ((x_size - solve_size) / 2) / p.sigma;
    let last = args.t_end <= args.t0 + delta_t;

    // If t0 + delta_t > t, fix that

    // FFT Solve center
    println!(
        "{}*FFT Solve, dt: {}, t1: {}",
        sp,
        delta_t,
        args.t0 + delta_t
    );

    let mut left_args = args;
    left_args.x_end = args.x0 + (p.sigma * delta_t);

    if last {
        println!("{}*Last", sp);
        return;
    }

    let mut new_args = args;
    new_args.t0 = args.t0 + delta_t;
    recursive_solve(new_args, p, level + 1);
}

fn main() {
    let p = FFTParams {
        sigma: 1,
        cutoff: 100,
        ratio: 0.5,
    };

    let args = FFTSolveArgs {
        boundaries: [Boundary::FIXED, Boundary::FIXED],
        t0: 0,
        t_end: 200,
        x0: 0,
        x_end: 200,
    };

    recursive_solve(args, p, 0);
}
