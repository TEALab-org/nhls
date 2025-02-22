use crate::domain::*;
use crate::time_varying::*;
use crate::util::*;

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
        layer: usize,
    ) -> CircStencil<GRID_DIMENSION> {
        println!("build_range: {} - {}", start_time, end_time);
        debug_assert!(end_time > start_time);

        // Two Base cases is combining to single step stencils
        if end_time - start_time == 2 {
            let mut s1 = CircStencil::new(self.stencil_slopes);
            s1.add_tv_stencil(self.stencil, start_time);

            let mut s2 = CircStencil::new(self.stencil_slopes);
            s2.add_tv_stencil(self.stencil, start_time + 1);

            let r = CircStencil::convolve(&s1, &s2);
            println!(
                " -- base 2, s: {}",
                r.domain.buffer().iter().sum::<f64>()
            );
            //write_debug_file(&format!("s{}e{}.txt", start_time, end_time), &r.domain);
            return r;
        }

        // Edgecase, not preferable
        if end_time - start_time == 1 {
            let mut s1 = CircStencil::new(self.stencil_slopes);
            s1.add_tv_stencil(self.stencil, start_time);
            //write_debug_file(&format!("s{}e{}.txt", start_time, end_time), &s1.domain);
            println!(
                " -- base 1, s: {}",
                s1.domain.buffer().iter().sum::<f64>()
            );
            return s1;
        }

        let mid = (start_time + end_time) / 2;
        let s1 = self.build_range(start_time, mid, layer + 1);
        let s2 = self.build_range(mid, end_time, layer + 1);
        let r = CircStencil::convolve(&s1, &s2);

        //write_debug_file(&format!("s{}e{}.txt", start_time, end_time), &r.domain);
        println!(" -- recurse, s: {}", r.domain.buffer().iter().sum::<f64>());
        r
    }
}
