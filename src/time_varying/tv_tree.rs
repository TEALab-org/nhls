use crate::domain::*;
use crate::time_varying::*;
use crate::util::*;
use fftw::array::*;
use std::collections::HashMap;

pub struct TVTree<
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> {
    pub stencil: &'a StencilType,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
    pub aabb: AABB<GRID_DIMENSION>,
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    > TVTree<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
{
    pub fn new(stencil: &'a StencilType, aabb: AABB<GRID_DIMENSION>) -> Self {
        let stencil_slopes = stencil.slopes();

        TVTree {
            stencil,
            stencil_slopes,
            aabb,
        }
    }

    pub fn build_range(
        &self,
        start_time: usize,
        end_time: usize,
    ) -> CircStencil<GRID_DIMENSION> {
        println!("build_range: {} - {}", start_time, end_time);
        debug_assert!(end_time > start_time);

        // Two Base cases is combining to single step stencils
        if end_time - start_time == 2 {
            let mut s1 = CircStencil::new(self.stencil_slopes);
            s1.add_tv_stencil(self.stencil, start_time);

            let mut s2 = CircStencil::new(self.stencil_slopes);
            s2.add_tv_stencil(self.stencil, start_time + 1);

            return CircStencil::convolve(&s1, &s2);
        }

        // Edgecase, not preferable
        if end_time - start_time == 1 {
            let mut s1 = CircStencil::new(self.stencil_slopes);
            s1.add_tv_stencil(self.stencil, start_time);
            return s1;
        }

        let mid = (start_time + end_time) / 2;
        let s1 = self.build_range(start_time, mid);
        let s2 = self.build_range(mid, end_time);

        CircStencil::convolve(&s1, &s2)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_build_range() {
        let aabb = AABB::new(matrix![0, 999]);
        let s = crate::standard_stencils::heat_1d(1.0, 1.0, 0.2);
        let t = TVTree::new(&s, aabb);
        t.build_range(0, 40);
    }
}

/*
pub type TVNodeId = usize;

pub struct SingleNode<const NEIGHBORHOOD_SIZE: usize> {
    weights: Values<NEIGHBORHOOD_SIZE>,
}

pub struct RangeNode<const GRID_DIMENSION: usize> {
    circ_stencil: CircStencil<GRID_DIMENSION>,
}

pub enum TVNode<const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize> {
    Single(SingleNode<NEIGHBORHOOD_SIZE>),
    Range(RangeNode<GRID_DIMENSION>),
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct RangeKey {
    step_min: usize,
    step_max: usize,
}

pub struct TVTree<
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> {
    stencil: &'a StencilType,
    node_map: HashMap<RangeKey, TVNode<GRID_DIMENSION, NEIGHBORHOOD_SIZE>>,
    stencil_slopes: Bounds<GRID_DIMENSION>,
}

impl <
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> TVTree<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType> {

    pub fn new(stencil: &'a StencilType) -> Self {
        let stencil_slopes = stencil.slopes();

        TVTree {
            stencil,
            stencil_slopes,
            node_map: HashMap::new(),
        }
    }

    pub fn build_range(&mut self, start_time: usize, end_time: usize) {
        if end_time - start_time == 1 {

        }


    }

    pub fn build_tree(&mut self, start_time: usize, end_time: usize) {
        // Add all the single weight ones in
        for t in start_time..end_time {
            let node = TVNode::Single(SingleNode {weights: self.stencil.weights(t)});
            let key = RangeKey {
                step_min: t,
                step_max: t + 1,
            };
            self.node_map.insert(key, node);
        }
    }


}
*/
