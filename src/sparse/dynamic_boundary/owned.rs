use crate::sparse::dynamic_boundary::*;

pub struct OwnedDynamicBoundary<const GRID_DIMENSION: usize> {
    inside: CoordSet<GRID_DIMENSION>,
    outside: CoordSet<GRID_DIMENSION>,
}

impl<const GRID_DIMENSION: usize> OwnedDynamicBoundary<GRID_DIMENSION> {
    pub fn new(
        inside: CoordSet<GRID_DIMENSION>,
        outside: CoordSet<GRID_DIMENSION>,
    ) -> Self {
        Self { inside, outside }
    }
}

impl<const GRID_DIMENSION: usize> DynamicBoundary<GRID_DIMENSION>
    for OwnedDynamicBoundary<GRID_DIMENSION>
{
    fn inside(&self) -> &CoordSet<GRID_DIMENSION> {
        &self.inside
    }

    fn outside(&self) -> &CoordSet<GRID_DIMENSION> {
        &self.outside
    }

    fn to_owned(self) -> Self {
        self
    }
}
