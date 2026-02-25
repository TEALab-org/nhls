use crate::sparse::dynamic_boundary::*;
use crate::stencil::*;

pub fn dilate_in<
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    InputBoundType: DynamicBoundary<GRID_DIMENSION>,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
>(
    boundary: InputBoundType,
    stencil: StencilType,
) -> OwnedDynamicBoundary<GRID_DIMENSION> {
    // Old in becomes out, calc new in?

    let mut new_in = CoordSet::empty();

    for coord in boundary.inside().coord_iter() {
        for offset in stencil.roi_offsets() {
            let n_coord = coord + offset;
            if !boundary.inside().contains(&n_coord)
                && !boundary.outside().contains(&n_coord)
            {
                new_in.add(n_coord);
            }
        }
    }

    OwnedDynamicBoundary::new(new_in, boundary.inside().clone())
}
