use crate::stencil::*;
use crate::util::*;
use core::f64;

pub fn heat_1d(dt: f64, dx: f64, k: f64) -> Stencil<1, 3> {
    Stencil::new([[1], [-1], [0]], move |args: &[f64; 3]| {
        let left = args[1];
        let middle = args[2];
        let right = args[0];
        middle + (k * dt / (dx * dx)) * (left - 2.0 * middle + right)
    })
}

pub fn heat_2d(dt: f64, dx: f64, dy: f64, k_x: f64, k_y: f64) -> Stencil<2, 5> {
    Stencil::new(
        [[1, 0], [0, -1], [-1, 0], [0, 1], [0, 0]],
        move |args: &[f64; 5]| {
            let middle = args[4];
            let left = args[2];
            let right = args[1];
            let bottom = args[1];
            let top = args[3];
            middle
                + (k_x * dt / (dx * dx)) * (left - 2.0 * middle + right)
                + (k_y * dt / (dy * dy)) * (top - 2.0 * middle + bottom)
        },
    )
}

pub fn heat_3d(
    dt: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    k_x: f64,
    k_y: f64,
    k_z: f64,
) -> Stencil<3, 7> {
    Stencil::new(
        [
            [0, 0, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ],
        move |args: &[f64; 7]| {
            let middle = args[0];
            let left = args[1];
            let right = args[2];
            let bottom = args[3];
            let top = args[4];
            let front = args[5];
            let back = args[6];
            middle
                + (k_x * dt / (dx * dx)) * (left - 2.0 * middle + right)
                + (k_y * dt / (dy * dy)) * (top - 2.0 * middle + bottom)
                + (k_z * dt / (dz * dz)) * (front - 2.0 * middle + back)
        },
    )
}

pub struct RotatingAdvectionStencil {
    offsets: [Coord<2>; 5],

    /// Steps per rotation
    frequency: f64,

    /// Hold central weight constant
    central_weight: f64,

    dist_mod: f64,
}

impl RotatingAdvectionStencil {
    pub fn new(frequency: f64, central_weight: f64) -> Self {
        let offsets = [
            vector![1, 0],
            vector![0, -1],
            vector![-1, 0],
            vector![0, 1],
            vector![0, 0],
        ];
        let dist_mod = 1.0 - central_weight;
        RotatingAdvectionStencil {
            offsets,
            frequency,
            central_weight,
            dist_mod,
        }
    }
}

impl TVStencil<2, 5> for RotatingAdvectionStencil {
    fn offsets(&self) -> &[Coord<2>; 5] {
        &self.offsets
    }

    // Model advection distribution with
    // *(0.5 sin(a + time * frequency) + 1.0) / 2Pi
    // That integrates to 1.0 over unit circle.
    // So each of the neighbors gets a quadrant,
    // i.e. integrate that for a in (0, Pi / 2) for quadrant 1
    // Did that with sympy
    // and got these equations
    fn weights(&self, global_time: usize) -> Values<5> {
        let f_gt = global_time as f64;
        let sqrt_2 = 2.0f64.sqrt();
        let pi = f64::consts::PI;
        let q1 = self.dist_mod
            * (sqrt_2 * (f_gt * self.frequency + pi / 4.0).sin() + pi)
            / (4.0 * pi);
        let q2 = self.dist_mod
            * (sqrt_2 * (f_gt * self.frequency + pi / 4.0).cos() + pi)
            / (4.0 * pi);
        let q3 = self.dist_mod
            * (-sqrt_2 * (f_gt * self.frequency + pi / 4.0).sin() + pi)
            / (4.0 * pi);
        let q4 = self.dist_mod
            * (-sqrt_2 * (f_gt * self.frequency + pi / 4.0).cos() + pi)
            / (4.0 * pi);

        vector![q1, q2, q3, q4, self.central_weight]
    }
}

pub struct TVHeat1D {
    offsets: [Coord<1>; 3],
}

impl TVHeat1D {
    pub fn new() -> Self {
        let offsets = [vector![1], vector![-1], vector![0]];
        TVHeat1D { offsets }
    }
}

impl TVStencil<1, 3> for TVHeat1D {
    fn offsets(&self) -> &[Coord<1>; 3] {
        &self.offsets
    }

    fn weights(&self, global_time: usize) -> Values<3> {
        let t_f = global_time as f64;
        let e = (-t_f * 0.01).exp();
        let cw = 1.0 - e;
        let nw = e / 2.0;
        vector![nw, nw, cw]
    }
}

pub struct TVHeat2D {
    offsets: [Coord<2>; 5],
}

impl TVHeat2D {
    pub fn new() -> Self {
        let offsets = [
            vector![1, 0],
            vector![0, -1],
            vector![-1, 0],
            vector![0, 1],
            vector![0, 0],
        ];
        TVHeat2D { offsets }
    }
}

impl TVStencil<2, 5> for TVHeat2D {
    fn offsets(&self) -> &[Coord<2>; 5] {
        &self.offsets
    }

    fn weights(&self, global_time: usize) -> Values<5> {
        let t_f = global_time as f64;
        let e = (-t_f * 0.01).exp();
        let cw = 1.0 - e;
        let nw = e / 5.0;
        vector![nw, nw, nw, nw, cw]
    }
}
