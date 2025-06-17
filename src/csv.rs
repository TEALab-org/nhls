use crate::domain::*;
use crate::util::*;
use std::io::prelude::*;

pub fn write_csv_2d<P: AsRef<std::path::Path>>(
    domain: &SliceDomain<2>,
    path: &P,
) {
    println!("Writing: {:?}", path.as_ref());
    // Open file
    let mut output =
        std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let aabb = domain.aabb();

    // Write line for each y value
    for y in aabb.bounds[(1, 0)]..=aabb.bounds[(1, 1)] {
        let r = domain.view(&vector![aabb.bounds[(0, 0)], y]);
        write!(output, "{r}").unwrap();
        for x in (aabb.bounds[(0, 0)] + 1)..=aabb.bounds[(0, 1)] {
            let mut c = Coord::zero();
            c[0] = x;
            c[1] = y;
            let r = domain.view(&c);
            //let b = (r > 2.0 * std::f64::EPSILON) as usize;
            write!(output, ", {r}").unwrap();
        }
        writeln!(output).unwrap();
    }
}
