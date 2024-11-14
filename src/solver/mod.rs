pub mod fft_plan;
pub mod periodic_plan;

// Axis Aligned subset of global state at time t
pub struct State<const GRID_DIMENSION: usize> {
    pub space_min: [usize; GRID_DIMENSION],
    pub space_max: [usize; GRID_DIMENSION],
    pub t: usize,
    pub values: Vec<f32>,
}
