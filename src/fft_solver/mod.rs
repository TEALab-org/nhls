pub type OpId = usize;
pub type NodeId = usize;

pub const MIN_ALIGNMENT: usize = 128;

mod convolution_op;
mod periodic_solver;
mod plan_type;

pub use convolution_op::*;
pub use periodic_solver::*;
pub use plan_type::*;
