// Used for Stencil traits
#![feature(trait_alias)]
// We use these alot with const ranges,
// don't like this warning for this codebase.
#![allow(clippy::needless_range_loop)]

pub mod decomposition;
pub mod domain;
pub mod image;
pub mod par_slice;
pub mod par_stencil;
pub mod solver;
pub mod stencil;
pub mod util;
pub mod image_1d_example;
