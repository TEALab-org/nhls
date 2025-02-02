use crate::util::*;

/// Calculate the input region for a frustrum solve
/// based on output region size and other parameters.
pub fn frustrum_input_aabb<const GRID_DIMENSION: usize>(
    steps: usize,
    output_box: &AABB<GRID_DIMENSION>,
    sloped_sides: &Bounds<GRID_DIMENSION>,
    stencil_slopes: &Bounds<GRID_DIMENSION>,
) -> AABB<GRID_DIMENSION> {
    let trapezoid_slopes =
        slopes_to_outward_diff(&stencil_slopes.component_mul(sloped_sides));
    output_box.add_bounds_diff(steps as i32 * trapezoid_slopes)
}

/// This needs to match logic from AABB::decomposition
pub fn decomposition_slopes<const DIMENSION: usize>(
) -> [[Bounds<DIMENSION>; 2]; DIMENSION] {
    let lower = 0;
    let upper = 1;

    // Start will all ones, turn off things that don't slope
    let mut result = [[Bounds::from_element(1); 2]; DIMENSION];

    for d in 0..DIMENSION {
        result[d][lower][(d, lower)] = 0;
        result[d][upper][(d, upper)] = 0;

        for d0 in d + 1..DIMENSION {
            result[d][lower][(d0, lower)] = 0;
            result[d][lower][(d0, upper)] = 0;
            result[d][upper][(d0, lower)] = 0;
            result[d][upper][(d0, upper)] = 0;
        }
    }

    result
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn decomposition_slopes_test() {
        {
            let d1 = decomposition_slopes::<1>();
            assert_eq!(d1, [[matrix![0, 1], matrix![1, 0]]]);
        }

        {
            let d2 = decomposition_slopes::<2>();
            let expected = [
                [matrix![0, 1; 0, 0], matrix![1, 0; 0, 0]],
                [matrix![1, 1; 0, 1], matrix![1, 1; 1, 0]],
            ];
            assert_eq!(d2, expected);
        }

        {
            let d3 = decomposition_slopes::<3>();
            let expected = [
                [matrix![0, 1; 0,0; 0,0], matrix![1, 0; 0, 0; 0, 0]],
                [matrix![1, 1; 0, 1; 0,0], matrix![1, 1; 1, 0; 0,0]],
                [matrix![1, 1; 1, 1; 0, 1], matrix![1, 1; 1, 1; 1, 0]],
            ];
            assert_eq!(d3, expected);
        }
    }
}
