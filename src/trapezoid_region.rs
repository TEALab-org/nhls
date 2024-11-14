use crate::stencil::*;
use crate::util::*;
use nalgebra;

pub struct Region<const GRID_DIMENSION: usize> {
    min_0: Bound<GRID_DIMENSION>,
    max_0: Bound<GRID_DIMENSION>,
    t: i32,
    slopes: Slopes<GRID_DIMENSION>,
}
/*
impl <const GRID_DIMENSION: usize> Region<GRID_DIMENSION> {
    fn new(end_min: Bound<GRID_DIMENSION>, end_max: Bound<GRID_DIMENSION>, t: i32) -> Self {

        Region {
            min_0,
            max_0,
            t,
            slopes,
        }
    }

    fn par_apply<
        NumType: NumTrait,
        const NEIGHBORHOOD_SIZE: usize,
        Operation,
    > (
        &self,
        stencil: &Stencil<NumType, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        input_buffer: &mut [NumType],
        output_buffer: &mut [NumType],
        chunk_size: usize
    ) where Operation: StencilOperation<NumType, NEIGHBORHOOD_SIZE>
     {
        debug_assert_eq!(input_buffer.len(), output_buffer.len());

        let mut a = input_buffer;
        let mut b = output_buffer;


        let mut min = self.min_0;
        let mut max = self.min_0;
        let mut previous_slice_dims = max - min;
        for _ in 0..self.t {
            min += self.slopes.column(0);
            max -= self.slopes.column(1);
            let time_slice_dims = max - min;

            let
        }
    }
}
*/
