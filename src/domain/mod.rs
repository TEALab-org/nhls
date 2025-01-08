//! This module has things for managing the domain,
//! which really means retreiving values based on world coordinates.
//! However, we will often work on buffers that represent a small
//! peice of the domain.
//! We use views to represent the peice of space a buffer represents,
//! and translate from world coordinates into view coordinates.

mod bc;
mod gather_args;
mod stack;
mod view;

pub use bc::*;
pub use gather_args::*;
pub use stack::*;
pub use view::*;
