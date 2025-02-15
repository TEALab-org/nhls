use crate::domain::*;
use crate::par_slice;
use crate::stencil;
use crate::time_varying::*;
use crate::util::*;
use fftw::array::*;
use fftw::plan::*;

pub struct CircStencil<const GRID_DIMENSION: usize> {
    pub slopes: Bounds<GRID_DIMENSION>,
    pub domain: OwnedDomain<GRID_DIMENSION>,
}

impl<const GRID_DIMENSION: usize> CircStencil<GRID_DIMENSION> {
    pub fn new(slopes: Bounds<GRID_DIMENSION>) -> Self {
        let domain = OwnedDomain::new(slopes_to_circ_aabb(&slopes));
        CircStencil { slopes, domain }
    }

    pub fn slopes(&self) -> Bounds<GRID_DIMENSION> {
        self.slopes
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

    pub fn to_offset_weights<'a>(
        &'a self,
    ) -> impl Iterator<Item = (Coord<GRID_DIMENSION>, f64)> + 'a {
        let total_width = self.slopes.column(0) + self.slopes.column(1);
        //let max: Coord<GRID_DIMENSION> = self.slopes.column(1);
        self.domain.aabb().coord_iter().map(move |domain_coord| {
            let weight = self.domain.view(&domain_coord);
            let mut offset = Coord::zero();
            for d in 0..GRID_DIMENSION {
                let d_max = total_width[d] + 1;
                let o = -(((domain_coord[d] + self.slopes[(d, 0)]) % d_max) - self.slopes[(d, 0)]);
                offset[d] = o;
            }
            //println!("domainc: {:?} -> {:?}, {:?}", domain_coord, offset, total_width);
            (offset, weight)
        })
    }

    pub fn add_circ_stencil(&mut self, other: &Self) {
        println!("  --- add_circ pre add size: {}", self.domain.aabb());
        self.add_offset_weights(other.to_offset_weights());
        println!("  --- add_circ post add sum: {}", self.domain.buffer().iter().sum::<f64>());
    }

    pub fn convolve(s1: &Self, s2: &Self) -> Self {
        let chunk_size = 10000;
        let c = s1.slopes() + s2.slopes();
        let aabb = slopes_to_circ_aabb(&c);

        // Create CircStencils of proper size
        let mut cs1 = CircStencil::new(c);
        cs1.add_circ_stencil(s1);
        let mut cs2 = CircStencil::new(c);
        cs2.add_circ_stencil(s2);

        // Setup FFT plans
        let size = aabb.exclusive_bounds();
        let plan_size = size.try_cast::<usize>().unwrap();
        let forward_plan = fftw::plan::R2CPlan64::aligned(
            plan_size.as_slice(),
            fftw::types::Flag::ESTIMATE,
        )
        .unwrap();
        let backward_plan = fftw::plan::C2RPlan64::aligned(
            plan_size.as_slice(),
            fftw::types::Flag::ESTIMATE,
        )
        .unwrap();

        // Forward pass on stencils
        let mut complex1: AlignedVec<c64> =
            AlignedVec::new(aabb.complex_buffer_size());
        forward_plan
            .r2c(cs1.domain.buffer_mut(), &mut complex1)
            .unwrap();
        let mut complex2: AlignedVec<c64> =
            AlignedVec::new(aabb.complex_buffer_size());
        forward_plan
            .r2c(cs2.domain.buffer_mut(), &mut complex2)
            .unwrap();

        println!(
            "  -- convolve: s1: {}, s2: {}, cs1: {}, cs2: {}",
            s1.domain.buffer().iter().sum::<f64>(),
            s2.domain.buffer().iter().sum::<f64>(),
            cs1.domain.buffer().iter().sum::<f64>(),
            cs2.domain.buffer().iter().sum::<f64>(),
        );

        // Convolve and backward pass
        par_slice::multiply_by(&mut complex1, &complex2, chunk_size);
        backward_plan
            .c2r(&mut complex1, cs1.domain.buffer_mut())
            .unwrap();
        let n_r = cs1.domain.aabb().buffer_size();
        par_slice::div(cs1.domain.buffer_mut(), n_r as f64, chunk_size);

        cs1
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn index_testing() {
        let slopes = matrix![2, 2];
        let s = CircStencil::new(slopes);
        let mut coord_set = std::collections::HashSet::new();
        for (o, _w) in s.to_offset_weights() {
            assert!(!coord_set.contains(&o));
            coord_set.insert(o);
            println!("Coord: {:?}", o);
        }
    }
}
