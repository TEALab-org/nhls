use crate::stencil::*;
use crate::util::*;

pub trait BCLookup<const GRID_DIMENSION: usize>: Sync {
    fn value(&self, coord: &Coord<GRID_DIMENSION>, input: &[f32]) -> f32;
}

pub fn gather_args<Lookup, Operation, const GRID_DIMENSION: usize, const NEIGHBORHOOD_SIZE: usize>(
    stencil: &StencilF32<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    lookup: &Lookup,
    input: &[f32],
    coord: &Coord<GRID_DIMENSION>,
) -> [f32; NEIGHBORHOOD_SIZE]
where
    Operation: StencilOperation<f32, NEIGHBORHOOD_SIZE>,
    Lookup: BCLookup<GRID_DIMENSION>,
{
    let mut result = [0.0; NEIGHBORHOOD_SIZE];
    for (i, n_i) in stencil.offsets().iter().enumerate() {
        let n_coord = coord + n_i;
        result[i] = lookup.value(&n_coord, input);
    }
    result
}

pub struct PeriodicBCLookup<const GRID_DIMENSION: usize> {
    bound: Coord<GRID_DIMENSION>,
}

impl<const GRID_DIMENSION: usize> PeriodicBCLookup<GRID_DIMENSION> {
    pub fn new(bound: Coord<GRID_DIMENSION>) -> Self {
        PeriodicBCLookup { bound }
    }
}

impl<const GRID_DIMENSION: usize> BCLookup<GRID_DIMENSION> for PeriodicBCLookup<GRID_DIMENSION> {
    fn value(&self, coord: &Coord<GRID_DIMENSION>, input: &[f32]) -> f32 {
        let p_coord = periodic_index(coord, &self.bound);
        let i = linear_index(&p_coord, &self.bound);
        input[i]
    }
}

pub struct ConstantBCLookup<const GRID_DIMENSION: usize> {
    value: f32,
    bound: Coord<GRID_DIMENSION>,
}

impl<const GRID_DIMENSION: usize> ConstantBCLookup<GRID_DIMENSION> {
    pub fn new(value: f32, bound: Coord<GRID_DIMENSION>) -> Self {
        ConstantBCLookup { value, bound }
    }
}

impl<const GRID_DIMENSION: usize> BCLookup<GRID_DIMENSION> for ConstantBCLookup<GRID_DIMENSION> {
    fn value(&self, coord: &Coord<GRID_DIMENSION>, input: &[f32]) -> f32 {
        for d in 0..GRID_DIMENSION {
            let c = coord[d];
            if c < 0 || c >= self.bound[d] {
                return self.value;
            }
        }
        let i = linear_index(coord, &self.bound);
        input[i]
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use nalgebra::vector;

    #[test]
    fn periodic_lookup_test() {
        {
            let bound = vector![10];
            let n_r = real_buffer_size(&bound);
            let mut buffer = fftw::array::AlignedVec::new(n_r);
            for i in 0..n_r {
                buffer.as_slice_mut()[i] = i as f32;
            }
            let lookup = PeriodicBCLookup::new(bound);
            for i in 0..n_r {
                let v = lookup.value(&vector![i as i32], buffer.as_slice());
                let e = i as f32;
                assert_approx_eq!(f32, v, e);
            }

            {
                let v = lookup.value(&vector![-1], buffer.as_slice());
                let e = 9.0;
                assert_approx_eq!(f32, v, e);
            }

            {
                let v = lookup.value(&vector![10], buffer.as_slice());
                let e = 0.0;
                assert_approx_eq!(f32, v, e);
            }
        }
    }

    #[test]
    fn constant_lookup_test() {
        let bound = vector![10];
        let n_r = real_buffer_size(&bound);
        let mut buffer = fftw::array::AlignedVec::new(n_r);
        for i in 0..n_r {
            buffer.as_slice_mut()[i] = i as f32;
        }
        let lookup = ConstantBCLookup::new(-1.0, bound);
        for i in 0..n_r {
            let v = lookup.value(&vector![i as i32], &buffer);
            let e = i as f32;
            assert_approx_eq!(f32, v, e);
        }

        {
            let v = lookup.value(&vector![-1], &buffer);
            let e = -1.0;
            assert_approx_eq!(f32, v, e);
        }

        {
            let v = lookup.value(&vector![10], &buffer);
            let e = -1.0;
            assert_approx_eq!(f32, v, e);
        }
    }

    #[test]
    fn gather_args_test_const() {
        let bound = vector![10, 10];
        let n_r = real_buffer_size(&bound);
        let mut buffer = fftw::array::AlignedVec::new(n_r);
        for i in 0..n_r {
            let coord = linear_to_coord(i, &bound);
            buffer.as_slice_mut()[i] = (coord[0] + 3 * coord[1]) as f32;
        }
        let lookup = ConstantBCLookup::new(-4.0, bound);
        let stencil = Stencil::new(
            [[0, -1], [0, 1], [1, 0], [-1, 0], [0, 0]],
            |_: &[f32; 5]| -1.0,
        );
        let r = gather_args(&stencil, &lookup, &buffer, &vector![9, 9]);
        let e = [
            (9 + 3 * 8) as f32,
            -4.0,
            -4.0,
            (8 + 3 * 9) as f32,
            (9 + 3 * 9) as f32,
        ];
        for n in 0..r.len() {
            assert_approx_eq!(f32, r[n], e[n]);
        }
    }

    #[test]
    fn gather_args_test_periodic() {
        let bound = vector![10, 10];
        let n_r = real_buffer_size(&bound);
        let mut buffer = fftw::array::AlignedVec::new(n_r);
        for i in 0..n_r {
            let coord = linear_to_coord(i, &bound);
            buffer.as_slice_mut()[i] = (coord[0] + 3 * coord[1]) as f32;
        }
        let lookup = PeriodicBCLookup::new(bound);
        let stencil = Stencil::new(
            [[0, -1], [0, 1], [1, 0], [-1, 0], [0, 0]],
            |_: &[f32; 5]| -1.0,
        );
        let r = gather_args(&stencil, &lookup, &buffer, &vector![9, 9]);
        let e = [
            (9 + 3 * 8) as f32,
            9.0,
            (3 * 9) as f32,
            (8 + 3 * 9) as f32,
            (9 + 3 * 9) as f32,
        ];
        for n in 0..r.len() {
            assert_approx_eq!(f32, r[n], e[n]);
        }
    }
}
