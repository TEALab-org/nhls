use clap::Parser;

#[derive(Parser)]
pub struct Cli {
    /// x is the value we're going to raise to some power.
    #[arg(short)]
    x: usize,

    /// exponent for x.
    #[arg(short)]
    exp: usize,
}

pub fn log2ceil(i: usize) -> usize {
    (usize::BITS - i.leading_zeros()) as usize
}

pub fn repeated_square_reference(exp0: usize, i: usize) -> usize {
    debug_assert!(i > 0);

    let mut exp = exp0;
    let mut squares = i;
    let mut result = 1;
    let n = log2ceil(exp0);
    let mut operation = String::from("1 * ");
    println!("exp: ({}, {:b}), n: {}", exp, exp, n);

    for i in 0..n - 1 {
        println!(
            "i: {}, exp: ({}, {:b}), squares: {}, result: {}",
            i, exp, exp, squares, result
        );
        // Is the least significant bit 1?
        if exp & 1 == 1 {
            operation += &format!("x^{} * ", (2usize).pow(i as u32));
            result *= squares;
            exp -= 1;
            println!(
                "  - Odd exponent, exp: ({}, {:b}), result: {}",
                exp, exp, result
            );
        }
        // Shift off bit and square
        exp /= 2;
        println!("  - div2 op: exp: ({}, {:b})", exp, exp);
        squares *= squares;
    }

    // This case handles the leading one of exp
    debug_assert!(exp == 1);
    operation += &format!("x^{}", (2usize).pow(n as u32 - 1));
    result *= squares;
    println!(
        "Last i: {}, exp: {}, squares: {} result: {}",
        n - 1,
        exp,
        squares,
        result
    );

    println!("final result: {}", result);
    println!("operation, n: {}: {}", n, operation);
    result
}

fn main() {
    let cli = Cli::parse();
    repeated_square_reference(cli.exp, cli.x);
    println!("test value: {}", cli.x.pow(cli.exp as u32));
}
