pub struct Stencil {
    offsets: Vec<(i32, i32)>,
}

impl Stencil {
    pub fn new(offsets: Vec<(i32, i32)>) -> Self {
        Self {
            offsets
        }
    }

    pub fn offsets(&self) -> impl Iterator<Item=(i32, i32)> + use<'_> {
        self.offsets.iter().copied()
    }

    pub fn roi_offsets(&self) -> impl Iterator<Item=(i32, i32)> + use<'_> {
        self.offsets.iter().map(|&(x, y)| (-x, -y))
    }

    pub fn radius(&self) -> i32 {
        let mut r = 0;
        for &(xd, yd) in &self.offsets {
            r = r.max(xd.abs());
            r = r.max(yd.abs());
        }
        r
    }
}

pub fn s_npt(r: i32) -> Stencil {
    let mut offsets = vec![(0, 0)];
    for r in 1..=r {
        offsets.push((r, 0));
        offsets.push((0, r));
        offsets.push((-r, 0));
        offsets.push((0, -r));
    }
    Stencil::new(offsets)
}

pub fn s_rad(r: i32) -> Stencil {
    let mut offsets = Vec::new();
    let r_sq = r * r;
    for x in -r..=r {
        for y in -r..=r {
            let nr_sq = x * x + y * y;
            if nr_sq <= r_sq {
               offsets.push((x, y)); 
            }
        }
    }

    Stencil::new(offsets)
}

pub fn s_offset_rad(r: i32, xd: i32, yd: i32) -> Stencil {
    let mut offsets = Vec::new();
    let r_sq = r * r;
    for xc in -r..=r {
        for yc in -r..=r {
            let x = xc + xd;
            let y = yc + yd;
            let nr_sq = x * x + y * y;
            if nr_sq <= r_sq {
               offsets.push((x, y)); 
            }
        }
    }

    Stencil::new(offsets)

}
