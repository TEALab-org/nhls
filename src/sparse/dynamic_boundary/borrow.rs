use crate::sparse::dynamic_boundary::*;

pub struct BorrowDynamicBoundary<'a, const GRID_DIMENSION: usize> {
    inside: &'a CoordSet<GRID_DIMENSION>,
    outside: &'a CoordSet<GRID_DIMENSION>,
}

impl<'a, const GRID_DIMENSION: usize>
    BorrowDynamicBoundary<'a, GRID_DIMENSION>
{
    pub fn new(
        inside: &'a CoordSet<GRID_DIMENSION>,
        outside: &'a CoordSet<GRID_DIMENSION>,
    ) -> Self {
        Self { inside, outside }
    }
}

impl<'a, const GRID_DIMENSION: usize> DynamicBoundary<GRID_DIMENSION>
    for BorrowDynamicBoundary<'a, GRID_DIMENSION>
{
    fn inside(&self) -> &CoordSet<GRID_DIMENSION> {
        self.inside
    }

    fn outside(&self) -> &CoordSet<GRID_DIMENSION> {
        self.outside
    }

    fn to_owned(self) -> OwnedDynamicBoundary<GRID_DIMENSION> {
        OwnedDynamicBoundary::new(self.inside.clone(), self.outside.clone())
    }
}
