use crate::util::*;

/// All stencils operations must provide an operation that adheres to this type
pub trait StencilOperation<NumType: NumTrait, const NEIGHBORHOOD_SIZE: usize> =
    Fn(&[NumType; NEIGHBORHOOD_SIZE]) -> NumType + Sync;

pub type StencilF32<Operation, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize> =
    Stencil<f32, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>;

/// Stencils are the combination of an operation and neighbors
pub struct Stencil<
    NumType: NumTrait,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<NumType, NEIGHBORHOOD_SIZE>,
{
    operation: Operation,
    offsets: [Coord<GRID_DIMENSION>; NEIGHBORHOOD_SIZE],
    num_type: std::marker::PhantomData<NumType>,
}

impl<NumType, Operation, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>
    Stencil<NumType, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    Operation: StencilOperation<NumType, NEIGHBORHOOD_SIZE>,
    NumType: NumTrait,
{
    pub fn new(offsets: [[i32; GRID_DIMENSION]; NEIGHBORHOOD_SIZE], operation: Operation) -> Self {
        Stencil {
            offsets: std::array::from_fn(|i| Coord::from_column_slice(&offsets[i])),
            operation,
            num_type: std::marker::PhantomData,
        }
    }

    /// For linear stencils, we can extract the weight for a neighbor
    /// by passing in 1.0 for that neighbor and 0.0 for the others.
    pub fn extract_weights(&self) -> [NumType; NEIGHBORHOOD_SIZE] {
        let mut weights = [NumType::zero(); NEIGHBORHOOD_SIZE];
        let mut arg_buffer = [NumType::zero(); NEIGHBORHOOD_SIZE];
        for n in 0..NEIGHBORHOOD_SIZE {
            arg_buffer[n] = NumType::one();
            weights[n] = (self.operation)(&arg_buffer);
            arg_buffer[n] = NumType::zero();
        }
        weights
    }

    pub fn offsets(&self) -> &[Coord<GRID_DIMENSION>; NEIGHBORHOOD_SIZE] {
        &self.offsets
    }

    pub fn slopes(&self) -> Box<GRID_DIMENSION> {
        let mut result = Box::zero();
        for neighbor in self.offsets {
            for d in 0..GRID_DIMENSION {
                let neighbor_d = neighbor[d];
                if neighbor_d > 0 {
                    result[(d, 1)] = result[(d, 1)].max(neighbor_d);
                } else {
                    result[(d, 0)] = result[(d, 0)].max(-neighbor_d);
                }
            }
        }
        result
    }

    pub fn apply(&self, args: &[NumType; NEIGHBORHOOD_SIZE]) -> NumType {
        (self.operation)(args)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use nalgebra::matrix;

    #[test]
    fn extract_weights() {
        {
            let s = Stencil::new([[1]], |args: &[f32; 1]| 2.0 * args[0]);
            let w = s.extract_weights()[0];
            assert_approx_eq!(f32, w, 2.0);
        }

        {
            let s = Stencil::new([[1], [2], [3]], |args: &[f32; 3]| {
                2.0 * args[0] + 3.0 * args[1] + 5.0 * args[2]
            });
            let w = s.extract_weights();
            assert_approx_eq!(f32, w[0], 2.0, ulps = 1);
            assert_approx_eq!(f32, w[1], 3.0, ulps = 1);
            assert_approx_eq!(f32, w[2], 5.0, ulps = 1);
        }
    }

    #[test]
    fn slopes() {
        {
            let s = Stencil::new([[1]], |args: &[f32; 1]| 2.0 * args[0]);
            let w = s.slopes();
            assert_eq!(w, matrix![0, 1]);
        }

        {
            let s = Stencil::new([[-1]], |args: &[f32; 1]| 2.0 * args[0]);
            let w = s.slopes();
            assert_eq!(w, matrix![1, 0]);
        }

        {
            let s = Stencil::new(
                [[-1, 0], [0, 0], [1, 0], [0, 2], [0, -3]],
                |args: &[f32; 5]| 2.0 * args[0] + args[1],
            );
            let w = s.slopes();
            assert_eq!(w, matrix![1, 1; 3, 2]);
        }
    }
}
