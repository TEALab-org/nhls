#![allow(clippy::module_inception)]
/// Artifact I haven't wanted to address yet
mod stencil;

mod circ_stencil;
mod tv_stencil;

pub mod standard_stencils;

pub use circ_stencil::*;
pub use stencil::*;
pub use tv_stencil::*;
