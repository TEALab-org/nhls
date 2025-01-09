use crate::util::*;

pub struct PeriodicSolveParams<const DIMENSION: usize> {
    pub stencil_slopes: Bounds<DIMENSION>,
    pub cutoff: i32,
    pub ratio: f64,
    pub max_steps: Option<usize>,
}

pub struct PeriodicSolve<const DIMENSION: usize> {
    output_aabb: AABB<DIMENSION>,
    steps: usize,
}

pub fn find_periodic_solve<const DIMENSION: usize>(
    input_aabb: &AABB<DIMENSION>,
    params: &PeriodicSolveParams<DIMENSION>,
) -> Option<PeriodicSolve<DIMENSION>> {
    if input_aabb.min_size_len() <= params.cutoff {
        return None;
    }

    let (steps, output_aabb) = input_aabb.shrink(
        params.ratio,
        params.stencil_slopes,
        params.max_steps,
    );

    println!(
        "Found fft solve, steps: {}, region: {:?}",
        steps, output_aabb
    );

    Some(PeriodicSolve { output_aabb, steps })
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
