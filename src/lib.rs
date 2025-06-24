// Used for Stencil traits
#![feature(trait_alias)]
// We use these alot with const ranges,
// don't like this warning for this codebase.
#![allow(clippy::needless_range_loop)]

pub mod ap_solver;
pub mod build_info;
pub mod csv;
pub mod direct_solver;
pub mod domain;
pub mod fft_solver;
pub mod image;
pub mod image_1d_example;
pub mod image_2d_example;
pub mod image_3d_example;
pub mod image_example_util;
pub mod initial_conditions;
pub mod mem_fmt;
pub mod mirror_domain;
pub mod par_slice;
pub mod par_stencil;
pub mod solver_interface;
pub mod stencil;
pub mod time_varying;
pub mod util;
pub mod vtk;

pub use solver_interface::*;
pub use stencil::standard_stencils;

/// Please call this first thing!
/// This function
///     * initializes Rayon's global threadpool with the specified number of threads,
///     * Ensures that FFTW3 uses Rayon's threadpool
/// It is important to do this prior to solver generation and execution.
pub fn init_threads(threads: usize) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    fftw::threading::init_threads_f64().unwrap();

    // This is setting a default value,
    // Not strictly necessary,
    // i.e. we should set this explicitly whenever planning
    fftw::threading::plan_with_nthreads_f64(threads);
}
