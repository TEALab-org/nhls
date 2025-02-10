use crate::domain::*;
use crate::time_varying::*;
use crate::util::*;
use crate::par_slice;
use fftw::plan::*;
use fftw::array::*;


pub fn fft_mul<const GRID_DIMENSION: usize>(
    s1: &DynamicLinearStencil<GRID_DIMENSION>,
    s2: &DynamicLinearStencil<GRID_DIMENSION>,
) {
    let slopes_1 = s1.slopes();
    let slopes_2 = s2.slopes();
    let c = slopes_1 + slopes_2;
    let c2: Coord<GRID_DIMENSION> = c.column(0) + c.column(1);
    let mut b: Bounds<GRID_DIMENSION> = Bounds::zero();
    b.set_column(1, &c2);
    let aabb = AABB::new(b);

    let chunk_size = 1000;

    // Create FFT plans
    let size = aabb.exclusive_bounds();
    let plan_size = size.try_cast::<usize>().unwrap();
    let forward_plan = fftw::plan::R2CPlan64::aligned(
            plan_size.as_slice(),
            fftw::types::Flag::ESTIMATE,
        )
        .unwrap();
        let backward_plan = fftw::plan::C2RPlan64::aligned(
            plan_size.as_slice(),
            fftw::types::Flag::ESTIMATE,
        )
        .unwrap();

   
        // Create FFT of s1
        let mut domain1 = OwnedDomain::new(aabb);
        for (offset, weight) in s1.offset_weights() {
            // I don't understand why, but we found that this mirroring operation
            // was necessary. I think it was in the paper.
            // TODO: Why is this the case?
            let rn_i: Coord<GRID_DIMENSION> = offset * -1;
            let periodic_coord = aabb.periodic_coord(&rn_i);
            domain1.set_coord(&periodic_coord, *weight);
        }
        let mut complex1: AlignedVec<c64> = AlignedVec::new(aabb.complex_buffer_size());
        forward_plan.r2c(domain1.buffer_mut(), &mut complex1).unwrap();

        // Create FFT of s2
        let mut domain2 = OwnedDomain::new(aabb);
        for (offset, weight) in s2.offset_weights() {
            // I don't understand why, but we found that this mirroring operation
            // was necessary. I think it was in the paper.
            // TODO: Why is this the case?
            let rn_i: Coord<GRID_DIMENSION> = offset * -1;
            let periodic_coord = aabb.periodic_coord(&rn_i);
            domain2.set_coord(&periodic_coord, *weight);
        }
        let mut complex2: AlignedVec<c64> = AlignedVec::new(aabb.complex_buffer_size());
        forward_plan.r2c(domain2.buffer_mut(), &mut complex2).unwrap();

        // multiply and get result
        par_slice::multiply_by(&mut complex1, &complex2, chunk_size);
        backward_plan.c2r(&mut complex1, domain1.buffer_mut()).unwrap();
        let n_r = aabb.buffer_size();
        par_slice::div(domain1.buffer_mut(), n_r as f64, chunk_size);
        println!("bounds: {}", aabb);
        println!("{:?}", domain1.buffer());
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn fft_mul_test_1d() {
        {
            let ss = crate::standard_stencils::heat_1d(1.0, 1.0, 0.3);
            let ds = DynamicLinearStencil::from_static_stencil(&ss);
            fft_mul(&ds, &ds);
            let rss = ds.naive_compose(&ds);
            println!("{:?}", rss.offset_weights());
        }
    }

    #[test]
    fn fft_mul_test_2d() {
        {
            let ss = crate::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);
            let ds = DynamicLinearStencil::from_static_stencil(&ss);
            fft_mul(&ds, &ds);
            let rss = ds.naive_compose(&ds);
            println!("{:?}", rss.offset_weights());
        }
    }
}
