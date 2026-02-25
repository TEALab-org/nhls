use crate::sparse::dynamic_boundary::*;
use crate::util::*;
use image::ImageReader;

pub fn region_from_image<P: AsRef<std::path::Path>>(path: &P) -> CoordSet<2> {
    let mut region = CoordSet::empty();

    let image = ImageReader::open(path).unwrap()
    .decode().unwrap().into_rgb8();

    for (x, y, p) in image.enumerate_pixels() {
        if p.0[0] < 15 {
            region.add(vector![x as i32, y as i32]);
        }
    }

    region
}
