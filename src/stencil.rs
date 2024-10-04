/// All stencils operations must provide an operation that adheres to this type
pub trait StencilOperation<FloatType: num::Float, const NEIGHBORHOOD_SIZE: usize> =
    Fn(&[FloatType; NEIGHBORHOOD_SIZE]) -> FloatType;

/// Stencils are the combination of an operation and neighbors
pub struct Stencil<
    FloatType: num::Float,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<FloatType, NEIGHBORHOOD_SIZE>,
{
    operation: Operation,
    offsets: [[i32; GRID_DIMENSION]; NEIGHBORHOOD_SIZE],
    float_type: std::marker::PhantomData<FloatType>,
}

impl<FloatType, Operation, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>
    Stencil<FloatType, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    Operation: StencilOperation<FloatType, NEIGHBORHOOD_SIZE>,
    FloatType: num::Float,
{
    pub fn new(offsets: [[i32; GRID_DIMENSION]; NEIGHBORHOOD_SIZE], operation: Operation) -> Self {
        Stencil {
            offsets,
            operation,
            float_type: std::marker::PhantomData,
        }
    }

    /// For linear stencils, we can extract the weight for a neighbor
    /// by passing in 1.0 for that neighbor and 0.0 for the others.
    pub fn extract_weights(&self) -> [FloatType; NEIGHBORHOOD_SIZE] {
        let mut weights = [FloatType::zero(); NEIGHBORHOOD_SIZE];
        let mut arg_buffer = [FloatType::zero(); NEIGHBORHOOD_SIZE];
        for n in 0..NEIGHBORHOOD_SIZE {
            arg_buffer[n] = FloatType::one();
            weights[n] = (self.operation)(&arg_buffer);
            arg_buffer[n] = FloatType::zero();
        }
        weights
    }

    pub fn offsets(&self) -> &[[i32; GRID_DIMENSION]; NEIGHBORHOOD_SIZE] {
        &self.offsets
    }

    pub fn apply(&self, args: &[FloatType; NEIGHBORHOOD_SIZE]) -> FloatType {
        (self.operation)(args)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use float_cmp::assert_approx_eq;

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
}
