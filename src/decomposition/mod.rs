use crate::util::*;

pub struct FFTSolveParams<const DIMENSION: usize> {
    pub slopes: Bounds<DIMENSION>,
    pub cutoff: i32,
    pub ratio: f64,
}

pub struct FFTSolve<const DIMENSION: usize> {
    pub solve_region: AABB<DIMENSION>,
    pub steps: usize,
}

pub fn try_fftsolve<const DIMENSION: usize>(
    bounds: &AABB<DIMENSION>,
    params: &FFTSolveParams<DIMENSION>,
    max_steps: Option<usize>,
) -> Option<FFTSolve<DIMENSION>> {
    if bounds.min_size_len() <= params.cutoff {
        return None;
    }

    let (steps, solve_region) = bounds.shrink(params.ratio, params.slopes, max_steps);

    Some(FFTSolve {
        solve_region,
        steps: steps as usize,
    })
}

/// This needs to match logic from AABB::decomposition
pub fn decomposition_slopes<const DIMENSION: usize>() -> [[Bounds<DIMENSION>; 2]; DIMENSION] {
    let lower = 0;
    let upper = 1;

    // Start will all ones, turn off things that don't slope
    let mut result = [[Bounds::from_element(1); 2]; DIMENSION];

    for d in 0..DIMENSION {
        result[d][lower][(d, lower)] = 0;
        result[d][upper][(d, upper)] = 0;

        for d0 in d + 1..DIMENSION {
            result[d][lower][(d0,lower)] = 0;
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
                [matrix![1, 1; 1, 1; 0, 1], matrix![1, 1; 1, 1; 1, 0]]
            ];
            assert_eq!(d3, expected);
        }
    }
}
