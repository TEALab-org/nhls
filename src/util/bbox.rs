/// Bounding Box, inclusive.
pub struct BBox<const GRID_DIMENSION: usize> {
    values: nalgebra::SMatrix<i32, { GRID_DIMENSION }, 2>,
}
