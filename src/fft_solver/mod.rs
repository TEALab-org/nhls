pub type OpId = usize;

mod convolution_gen;
mod convolution_op;
mod convolution_store;
mod periodic_solver;
mod plan_type;

pub use convolution_gen::*;
pub use convolution_op::*;
pub use convolution_store::*;
pub use periodic_solver::*;
pub use plan_type::*;
