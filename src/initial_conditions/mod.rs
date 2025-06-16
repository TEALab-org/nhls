mod generate_ic;
pub mod normal_impulse;
pub mod rand;
use clap::ValueEnum;
pub use generate_ic::*;

#[derive(Copy, Clone, Debug, Default)]
pub enum ICType {
    #[default]
    Zero,
    Rand {
        max_val: i32,
    },
    Impulse {
        variance: f64,
    },
}

#[derive(Copy, Clone, Debug, ValueEnum, Default)]
pub enum ClapICType {
    #[default]
    /// All grid values are left at 0.0
    Zero,

    /// Random values between 0.0 and ic_dial
    Rand,

    /// Normal impulse with ic_dial variance
    Impulse,
}

impl ClapICType {
    pub fn to_ic_type(&self, dial: Option<f64>) -> ICType {
        let dial_v = dial.unwrap_or(0.0);
        match self {
            ClapICType::Zero => ICType::Zero,
            ClapICType::Rand => ICType::Rand {
                max_val: dial_v as i32,
            },
            ClapICType::Impulse => ICType::Impulse { variance: dial_v },
        }
    }
}
