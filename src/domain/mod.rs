//! This module has things for managing the domain,
//! which really means retrieving values based on world coordinates.
//! However, we will often work on buffers that represent a small
//! piece of the domain.
//! We use views to represent the piece of space a buffer represents,
//! and translate from world coordinates into view coordinates.

mod bc;
mod gather_args;
mod view;

pub use bc::*;
pub use gather_args::*;
pub use view::*;
