mod borrow;
mod boundary_set;
mod dilate;
mod owned;
mod vtk;

pub use borrow::*;
pub use boundary_set::*;
pub use dilate::*;
pub use owned::*;
pub use vtk::*;

pub trait DynamicBoundary<const GRID_DIMENSION: usize> {
    fn inside(&self) -> &BoundarySet<GRID_DIMENSION>;
    fn outside(&self) -> &BoundarySet<GRID_DIMENSION>;
    fn to_owned(self) -> OwnedDynamicBoundary<GRID_DIMENSION>;
}
