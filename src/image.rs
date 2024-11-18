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
