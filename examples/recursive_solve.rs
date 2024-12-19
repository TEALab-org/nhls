// Recursive domain decomposition.
//
// We need boundary condition functions
//
// What is value out of bounds

//
// Face vectors

/*
struct BoundaryFrustrum {
    x_start: i32,
    x_end: i32,
    bc_x_pos: bool,
    bc_y_pos: bool,
}

struct FFTOperation {
    in_x_start: i32,
    in_x_end: i32,
    out_x_start: i32,
    out_x_end: i32,
    steps: i32,
}

struct Recursive
*/

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
    // 1 for Yes (no boundary condition)
    // 0 for vertical (boundary condition)
    sloped: [usize; 2],

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

pub fn recursive_solve(
    args: FFTSolveArgs,
    p: FFTParams,
    level: usize,
    step: usize,
) {
    let sp = space(level);
    println!(
        "{}RS: l: {}, s: {}, args: {:?}, p: {:?}",
        sp, level, step, args, p
    );

    // Are we below cutoff?
    let x_size = args.x_end - args.x0;
    if x_size < p.cutoff {
        println!("{}* Naive solve to {}", sp, args.t_end);
        return;
    }

    // Compute FFT Solve
    // Should be even so
    let mut solve_size = {
        let mut solve_size = (p.ratio * x_size as f32) as usize;
        if solve_size % 2 != 0 {
            solve_size -= 1
        }
        solve_size
    };

    // Otherwise find delta t to end
    let mut delta_t = ((x_size - solve_size) / 2) / p.sigma;
    let last = args.t_end <= args.t0 + delta_t;
    if last {
        delta_t = args.t_end - args.t0;
        solve_size = x_size - 2 * (p.sigma * delta_t);
    }

    // FFT Solve center
    println!(
        "{}*FFT Solve, dt: {}, t1: {}, solve_size: {}",
        sp,
        delta_t,
        args.t0 + delta_t,
        solve_size,
    );

    // left recursion
    if args.sloped[0] == 0 {}

    let mut left_args = args;
    left_args.x_end = args.x0 + (p.sigma * delta_t);

    if last {
        println!("{}*Last", sp);
        return;
    }

    let mut new_args = args;
    new_args.t0 = args.t0 + delta_t;
    recursive_solve(new_args, p, level, step + 1);
}

fn main() {
    let p = FFTParams {
        sigma: 1,
        cutoff: 100,
        ratio: 0.5,
    };

    let args = FFTSolveArgs {
        //boundaries: [Boundary::FIXED, Boundary::FIXED],
        sloped: [0, 1],
        t0: 0,
        t_end: 212,
        x0: 0,
        x_end: 200,
    };

    recursive_solve(args, p, 0, 0);
}
