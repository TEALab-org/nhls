use crate::domain::Domain;
use crate::util::*;

pub struct Image1D {
    img_buffer: image::RgbImage,
}

impl Image1D {
    pub fn new(bound: Box<1>, lines: u32) -> Self {
        debug_assert_eq!(bound[(0, 0)], 0);
        Image1D {
            img_buffer: image::RgbImage::new(bound[(0, 1)] as u32 + 1, lines),
        }
    }

    pub fn add_line(&mut self, l: u32, v: &[f32]) {
        debug_assert!(l < self.img_buffer.height());
        debug_assert_eq!(v.len(), self.img_buffer.width() as usize);
        let gradient = colorous::TURBO;
        for x in 0..self.img_buffer.width() {
            let r = v[x as usize];
            let c = gradient.eval_continuous(r as f64);
            self.img_buffer.put_pixel(x, l, image::Rgb(c.as_array()));
        }
    }

    pub fn write(self, s: &str) {
        self.img_buffer.save(s).expect("Couldn't save image");
    }
}

pub fn image2d(domain: &Domain<2>, s: &str) {
    let view_box = domain.view_box();
    let diff = (view_box.column(1) - view_box.column(0)).add_scalar(1);
    let gradient = colorous::TURBO;
    let mut img = image::RgbImage::new(diff[0] as u32, diff[1] as u32);
    for l in 0..diff[0] * diff[1] {
        let coord = linear_to_coord_in_box(l as usize, domain.view_box());
        let r = domain.view(&coord);
        let c = gradient.eval_continuous(r as f64);
        img.put_pixel(coord[0] as u32, coord[1] as u32, image::Rgb(c.as_array()));
    }
    img.save(s).expect("Couldn't save image");
}
