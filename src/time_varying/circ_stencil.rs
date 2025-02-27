use crate::domain::*;
use crate::par_slice;
use crate::time_varying::*;
use crate::util::*;

pub struct CircStencil<'a, const GRID_DIMENSION: usize> {
    pub slopes: Bounds<GRID_DIMENSION>,
    pub domain: SliceDomain<'a, GRID_DIMENSION>,
}

impl<'a, const GRID_DIMENSION: usize> CircStencil<'a, GRID_DIMENSION> {
    pub fn new(slopes: Bounds<GRID_DIMENSION>, buffer: &'a mut [f64]) -> Self {
        let domain = SliceDomain::new(slopes_to_circ_aabb(&slopes), buffer);
        CircStencil { slopes, domain }
    }

    pub fn slopes(&self) -> Bounds<GRID_DIMENSION> {
        self.slopes
    }

    pub fn clear(&mut self) {
        par_slice::set_value(self.domain.buffer_mut(), 0.0, 10000);
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
        //println!("    --- view: {:?} -> {}, s: {}", periodic_coord, self.domain.view(&periodic_coord), self.domain.buffer().iter().sum::<f64>());
    }

    pub fn add_tv_stencil<
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    >(
        &mut self,
        stencil: &StencilType,
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

    pub fn to_offset_weights(
        &'a self,
    ) -> impl Iterator<Item = (Coord<GRID_DIMENSION>, f64)> + 'a {
        let total_width = self.slopes.column(0) + self.slopes.column(1);
        //let max: Coord<GRID_DIMENSION> = self.slopes.column(1);
        self.domain.aabb().coord_iter().map(move |domain_coord| {
            let weight = self.domain.view(&domain_coord);
            let mut offset = Coord::zero();
            for d in 0..GRID_DIMENSION {
                let d_max = total_width[d] + 1;
                let o = -(((domain_coord[d] + self.slopes[(d, 0)]) % d_max)
                    - self.slopes[(d, 0)]);
                offset[d] = o;
            }
            //println!("domainc: {:?} -> {:?}, {:?}", domain_coord, offset, total_width);
            (offset, weight)
        })
    }

    pub fn add_circ_stencil(&mut self, other: &Self) {
        //println!("  --- add_circ pre add size: {}", self.domain.aabb());
        self.add_offset_weights(other.to_offset_weights());
        /*
        println!(
            "  --- add_circ post add sum: {}",
            self.domain.buffer().iter().sum::<f64>()
        );
        */
    }
}
