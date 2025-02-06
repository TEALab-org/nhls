use crate::util::*;

/// For linear stencils, we can extract the weight for a neighbor
/// by passing in 1.0 for that neighbor and 0.0 for the others.
pub fn extract_weights<
    const NEIGHBORHOOD_SIZE: usize,
    F: Fn(&[f64; NEIGHBORHOOD_SIZE]) -> f64,
>(
    f: F,
) -> Values<NEIGHBORHOOD_SIZE> {
    let mut weights = Values::zero();
    let mut arg_buffer = [0.0; NEIGHBORHOOD_SIZE];
    for n in 0..NEIGHBORHOOD_SIZE {
        arg_buffer[n] = 1.0;
        weights[n] = f(&arg_buffer);
        arg_buffer[n] = 0.0;
    }
    weights
}

/// For this code base we only deal with linear stencils.
/// We view linear stencils as a combination of neighbor offsets and weights.
pub struct Stencil<const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>
{
    pub weights: Values<NEIGHBORHOOD_SIZE>,
    pub offsets: [Coord<GRID_DIMENSION>; NEIGHBORHOOD_SIZE],
}

impl<const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>
    Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>
{
    pub fn new<F: Fn(&[f64; NEIGHBORHOOD_SIZE]) -> f64>(
        offsets: [[i32; GRID_DIMENSION]; NEIGHBORHOOD_SIZE],
        operation: F,
    ) -> Self {
        let weights = extract_weights(operation);
        Stencil {
            offsets: std::array::from_fn(|i| {
                Coord::from_column_slice(&offsets[i])
            }),
            weights,
        }
    }

    pub fn weights(&self) -> &Values<NEIGHBORHOOD_SIZE> {
        &self.weights
    }

    pub fn offsets(&self) -> &[Coord<GRID_DIMENSION>; NEIGHBORHOOD_SIZE] {
        &self.offsets
    }

    pub fn slopes(&self) -> Bounds<GRID_DIMENSION> {
        let mut result = Bounds::zero();
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

    pub fn apply(&self, args: &Values<NEIGHBORHOOD_SIZE>) -> f64 {
        self.weights.component_mul(args).sum()
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
            let s = Stencil::new([[1]], |args: &[f64; 1]| 2.0 * args[0]);
            let w = s.weights()[0];
            assert_approx_eq!(f64, w, 2.0);
        }

        {
            let s = Stencil::new([[1], [2], [3]], |args: &[f64; 3]| {
                2.0 * args[0] + 3.0 * args[1] + 5.0 * args[2]
            });
            let w = s.weights();
            assert_approx_eq!(f64, w[0], 2.0, ulps = 1);
            assert_approx_eq!(f64, w[1], 3.0, ulps = 1);
            assert_approx_eq!(f64, w[2], 5.0, ulps = 1);
        }
    }

    #[test]
    fn slopes() {
        {
            let s = Stencil::new([[1]], |args: &[f64; 1]| 2.0 * args[0]);
            let w = s.slopes();
            assert_eq!(w, matrix![0, 1]);
        }

        {
            let s = Stencil::new([[-1]], |args: &[f64; 1]| 2.0 * args[0]);
            let w = s.slopes();
            assert_eq!(w, matrix![1, 0]);
        }

        {
            let s = Stencil::new(
                [[-1, 0], [0, 0], [1, 0], [0, 2], [0, -3]],
                |args: &[f64; 5]| 2.0 * args[0] + args[1],
            );
            let w = s.slopes();
            assert_eq!(w, matrix![1, 1; 3, 2]);
        }
    }
}
