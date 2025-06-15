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
    Zero,
    Rand,
    Impulse,
}

impl ClapICType {
    pub fn to_ic_type(&self, dial: f64) -> ICType {
        match self {
            ClapICType::Zero => ICType::Zero,
            ClapICType::Rand => ICType::Rand {
                max_val: dial as i32,
            },
            ClapICType::Impulse => ICType::Impulse { variance: dial },
        }
    }
}
