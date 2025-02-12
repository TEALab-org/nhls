use crate::domain::*;
use crate::stencil;
use crate::time_varying::*;
use crate::util::*;

pub struct CircStencil<const GRID_DIMENSION: usize> {
    slopes: Bounds<GRID_DIMENSION>,
    domain: OwnedDomain<GRID_DIMENSION>,
}

impl<const GRID_DIMENSION: usize> CircStencil<GRID_DIMENSION> {
    pub fn new(slopes: Bounds<GRID_DIMENSION>) -> Self {
        let total_width: Coord<GRID_DIMENSION> =
            slopes.column(0) + slopes.column(1);
        let mut domain_bounds: Bounds<GRID_DIMENSION> = Bounds::zero();
        domain_bounds.set_column(1, &total_width);
        let domain = OwnedDomain::new(AABB::new(domain_bounds));
        CircStencil { slopes, domain }
    }

    pub fn add_offset_weight(
        &mut self,
        offset: Coord<GRID_DIMENSION>,
        weight: f64,
    ) {
        // I don't understand why, but we found that this mirroring operation
        // was necessary. I think it was in the paper.
        // TODO: Why is this the case?
        let rn_i: Coord<GRID_DIMENSION> = offset * -1;
        let periodic_coord = self.domain.aabb().periodic_coord(&rn_i);
        self.domain.set_coord(&periodic_coord, weight);
    }

    pub fn add_tv_stencil<
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    >(
        &mut self,
        stencil: StencilType,
        global_time: usize,
    ) {
        let weights = stencil.weights(global_time);
        for i in 0..NEIGHBORHOOD_SIZE {
            let offset = stencil.offsets()[i];
            let weight = weights[i];
            self.add_offset_weight(offset, weight);
        }
    }

    pub fn add_offset_weights(
        &mut self,
        offset_weights: impl Iterator<Item = (Coord<GRID_DIMENSION>, f64)>,
    ) {
        for (offset, weight) in offset_weights {
            self.add_offset_weight(offset, weight);
        }
    }

    pub fn from_dynamic_stencil(
        &mut self,
        ds: &DynamicLinearStencil<GRID_DIMENSION>,
    ) {
        for (offset, weight) in ds.offset_weights() {
            self.add_offset_weight(*offset, *weight);
        }
    }

    pub fn to_offset_weights<'a>(
        &'a self,
    ) -> impl Iterator<Item = (Coord<GRID_DIMENSION>, f64)> + 'a {
        let total_width = self.slopes.column(0) + self.slopes.column(1);
        self.domain.aabb().coord_iter().map(move |domain_coord| {
            let weight = self.domain.view(&domain_coord);
            let mut offset = Coord::zero();
            for d in 0..GRID_DIMENSION {
                let d_max = total_width[d];
                let o = -((domain_coord[d] + d_max) % d_max) - d_max;
                offset[d] = o;
            }
            (offset, weight)
        })
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn index_testing() {
        let min: i32 = -1;
        let max = 2;
        let c = min.abs() + max + 1;
        println!("min: {}, max: {}, c: {}", min, max, c);
        for i in 0..c {
            let r = ((i + max) % c) - max;
            println!("i: {}, r: {}", i, -r);
        }
    }
}
