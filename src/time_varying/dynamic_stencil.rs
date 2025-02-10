use std::collections::HashMap;
use crate::stencil::*;
use crate::util::*;

/// Like base stencil class, but with dynamic neighborhood size
pub struct DynamicLinearStencil<const GRID_DIMENSION: usize> {
    pub offset_weights: Vec<(Coord<GRID_DIMENSION>, f64)>,
}

impl<const GRID_DIMENSION: usize>
    DynamicLinearStencil<GRID_DIMENSION>
{
    pub fn new(offset_weights: Vec<(Coord<GRID_DIMENSION>, f64)>) -> Self {
        DynamicLinearStencil { offset_weights }
    }

    pub fn offset_weights(&self) -> &[(Coord<GRID_DIMENSION>, f64)] {
        &self.offset_weights
    }

    pub fn slopes(&self) -> Bounds<GRID_DIMENSION> {
        let mut result = Bounds::zero();
        for (offset, _) in &self.offset_weights {
            for d in 0..GRID_DIMENSION {
                let neighbor_d = offset[d];
                if neighbor_d > 0 {
                    result[(d, 1)] = result[(d, 1)].max(neighbor_d);
                } else {
                    result[(d, 0)] = result[(d, 0)].max(-neighbor_d);
                }
            }
        }
        result
    }

    pub fn naive_compose(&self, other: &Self) -> Self {
        let mut offset_map: HashMap<Coord<GRID_DIMENSION>, f64> =
            HashMap::new();

        for (self_offset, self_weight) in &self.offset_weights {
            for (other_offset, other_weight) in &other.offset_weights {
                let offset = self_offset + other_offset;
                let weight = self_weight * other_weight;
                if let Some(current_weight) = offset_map.get_mut(&offset) {
                    *current_weight += weight;
                } else {
                    offset_map.insert(offset, weight);
                }
            }
        }

        let offset_weights: Vec<(Coord<GRID_DIMENSION>, f64)> =
            offset_map.drain().collect();
        DynamicLinearStencil { offset_weights }
    }

    pub fn from_static_stencil<const NEIGHBORHOOD_SIZE: usize>(
        other: &Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    ) -> Self {
        let mut offset_weights = Vec::with_capacity(NEIGHBORHOOD_SIZE);
        for n in 0..NEIGHBORHOOD_SIZE {
            offset_weights.push((other.offsets()[n], other.weights()[n]));
        }
        DynamicLinearStencil { offset_weights }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn naive_compose_test() {
        {
            let ss = Stencil::new([[-1]], |args| args[0]);
            let ds = DynamicLinearStencil::from_static_stencil(&ss);
            let rs = ds.naive_compose(&ds);
            assert_eq!(rs.offset_weights().len(), 1);
            assert_eq!(rs.offset_weights()[0], (vector![-2], 1.0))
        }
    
        {
            let ss = crate::standard_stencils::heat_1d(1.0, 1.0, 0.3);
            let ds = DynamicLinearStencil::from_static_stencil(&ss);
            let rs = ds.naive_compose(&ds);
            println!("{:?}", rs.offset_weights());
            assert_eq!(rs.offset_weights().len(), 5);
            assert_eq!(rs.slopes(), matrix![2, 2]);
        }
    }

}
