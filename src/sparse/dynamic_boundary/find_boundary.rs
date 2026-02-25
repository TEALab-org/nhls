use crate::sparse::dynamic_boundary::*;
use crate::stencil::*;

pub fn find_region_boundaries<
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
>(
    region: &CoordSet<GRID_DIMENSION>,
    stencil: StencilType
) -> OwnedDynamicBoundary<GRID_DIMENSION> {
    let mut inside = CoordSet::empty();
    let mut outside = CoordSet::empty();

    for coord in region.coord_iter() {
       for offset in stencil.offsets() {
            let n_coord = coord + offset;
            if !region.contains(&n_coord) {
                outside.add(n_coord);
            }
       }
    }

    for coord in outside.coord_iter() {
        for offset in stencil.roi_offsets() {
            let n_coord = coord + offset;
            if region.contains(&n_coord) {
                inside.add(n_coord);
            }
        }
    }

    OwnedDynamicBoundary::new(inside, outside)
}
