use crate::domain::*;
use crate::util::*;
use std::io::prelude::*;

pub fn write_debug_file<
    P: AsRef<std::path::Path>,
    const GRID_DIMENSION: usize,
    DomainType: DomainView<GRID_DIMENSION>,
>(
    path: &P,
    domain: &DomainType,
) {
    // Open file
    let mut output =
        std::io::BufWriter::new(std::fs::File::create(path).unwrap());

    // Write bounds line 1
    let aabb = domain.aabb();
    writeln!(output, "{aabb}").unwrap();

    // Write line for each y value
    for y in aabb.bounds[(1, 0)]..=aabb.bounds[(1, 1)] {
        write!(output, "*{y}: ").unwrap();
        for x in aabb.bounds[(0, 0)]..=aabb.bounds[(0, 1)] {
            let mut c = Coord::zero();
            c[0] = x;
            c[1] = y;
            let r = domain.view(&c);
            //let b = (r > 2.0 * std::f64::EPSILON) as usize;
            write!(output, "{r:.05}, ").unwrap();
        }
        writeln!(output).unwrap();
    }
}
