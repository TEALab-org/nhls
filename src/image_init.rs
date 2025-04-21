pub use crate::util::*;
pub use crate::domain::*;
use image::open;

// function takes greyscale png
// returns
// struct with
// AABB
// Two Owned Domains

pub struct ImageProb {
    pub aabb: AABB<2>,
    pub input: OwnedDomain<2>,
    pub output: OwnedDomain<2>,
}

impl ImageProb {
    pub fn load<P: AsRef<std::path::Path>>(path: &P, chunk_size: usize) -> ImageProb {
        // Load image
        let img = open(path).unwrap().into_luma16();
        let width = img.width() as i32 - 1;
        let height = img.height() as i32 - 1;
        println!("Loaded {} x {} image", width + 1, height + 1);

        // Get dimensions
        let aabb = AABB::new(matrix![0, width; 0, height]);

        // Create domains
        let mut input = OwnedDomain::new(aabb);
        let output = OwnedDomain::new(aabb);

        // Init input domain
        input.par_set_values(|coord| {
            let x = coord[0] as u32;
            let y = coord[1] as u32;
            let p = img.get_pixel(x, y).0[0];
            let v = p as f64 / u16::MAX as f64;
            //println!("p: {}, v: {}", p, v);
            v
        }, chunk_size);
        
        ImageProb {
            aabb,
            input,
            output,
        }
    }
}
