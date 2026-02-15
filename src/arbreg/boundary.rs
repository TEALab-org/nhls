use crate::arbreg::Region;
use crate::arbreg::Stencil;

pub fn find_region_boundaries_s(region: &Region, stencil: &Stencil) -> (Region, Region) {
    let mut inside_boundary = Region::empty();
    let mut outside_boundary = Region::empty();

    for &(x, y) in region.cell_iter() {
        for (xd, yd) in stencil.offsets() {
            if !region.contains(x + xd, y + yd) {
                outside_boundary.add(x + xd, y + yd);
            }
        }
    }

    for &(x, y) in outside_boundary.cell_iter() {
        for (xd, yd) in stencil.roi_offsets() {
            if region.contains(x + xd, y + yd) {
                inside_boundary.add(x + xd, y + yd);
            }
        }
    }

    // For all cells not in domain
    // if ROI 

    (inside_boundary, outside_boundary)
}

pub fn dilate_s(domain: &Region, stencil: &Stencil) -> Region {
    let mut remove = Vec::new();
    for &(x, y) in domain.cell_iter() {
        for (xd, yd) in stencil.offsets() {
            if !domain.contains(x + xd, y + yd) {
                remove.push((x, y));
                continue;
            }
        }
    }
    let mut n = domain.clone();
    for &(x, y) in &remove {
        n.remove(x, y);
    }
    n
}

pub fn find_region_boundaries(region: &Region) -> (Region, Region) {
    let mut inside_boundary = Region::empty();
    let mut outside_boundary = Region::empty();

    for &(x, y) in region.cell_iter() {
        if !region.contains(x + 1, y) {
            inside_boundary.add(x, y);
            outside_boundary.add(x + 1, y);
        }

        if !region.contains(x, y + 1) {
            inside_boundary.add(x, y);
            outside_boundary.add(x, y + 1);
        }

        if !region.contains(x - 1, y) {
            inside_boundary.add(x, y);
            outside_boundary.add(x - 1, y);
        }

        if !region.contains(x, y - 1) {
            inside_boundary.add(x, y);
            outside_boundary.add(x, y - 1);
        }
    }

    (inside_boundary, outside_boundary)
}

pub fn dilate(outside: &Region, boundary: &Region) -> Region {
    let mut inside = Region::empty();

    for &(x, y) in boundary.cell_iter() {
        for offs in [[1, 0], [0, 1], [-1, 0], [0, -1]] {
            let xp = x + offs[0];
            let yp = y + offs[1];
            if !outside.contains(xp, yp) && !boundary.contains(xp, yp) {
                inside.add(xp, yp);
            }
        }
    }

    inside
}

pub fn dilate_n(outside: &Region, boundary: &Region, n: usize) -> (Region, Region) {
    let mut i1 = outside.clone();
    let mut i2 = boundary.clone();
    for _ in 0..n {
        let i3 = dilate(&i1, &i2);
        i1 = i2;
        i2 = i3;
    }
    (i1, i2)
}

pub fn fill_boundary(outside: &Region, boundary: &Region) -> Region {
    let mut filled = Region::empty();
    let mut new_points = boundary.clone(); 
    let mut next_points = Region::empty();
    while !new_points.is_empty() {
        for &(x, y) in new_points.cell_iter() {
            filled.add(x, y);
            for offs in [[1, 0], [0, 1], [-1, 0], [0, -1]] {
                let xp = x + offs[0];
                let yp = y + offs[1];
                if !outside.contains(xp, yp) && !filled.contains(xp, yp) {
                    next_points.add(xp, yp);
                }
            }
        }
        std::mem::swap(&mut new_points, &mut next_points);
        next_points.clear();
    }

    filled
}
