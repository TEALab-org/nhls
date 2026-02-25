mod borrow;
mod coord_set;
mod dilate;
mod owned;
mod vtk;

pub use borrow::*;
pub use coord_set::*;
pub use dilate::*;
pub use owned::*;
pub use vtk::*;

pub trait DynamicBoundary<const GRID_DIMENSION: usize> {
    fn inside(&self) -> &CoordSet<GRID_DIMENSION>;
    fn outside(&self) -> &CoordSet<GRID_DIMENSION>;
    fn to_owned(self) -> OwnedDynamicBoundary<GRID_DIMENSION>;
}
