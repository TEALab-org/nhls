use crate::stencil::*;

pub fn heat_1d(
    dt: f64,
    dx: f64,
    k: f64,
) -> StencilF64<impl StencilOperation<f64, 3>, 1, 3> {
    Stencil::new([[-1], [0], [1]], move |args: &[f64; 3]| {
        let left = args[0];
        let middle = args[1];
        let right = args[2];
        middle + (k * dt / (dx * dx)) * (left - 2.0 * middle + right)
    })
}

pub fn heat_2d(
    dt: f64,
    dx: f64,
    dy: f64,
    k_x: f64,
    k_y: f64,
) -> StencilF64<impl StencilOperation<f64, 5>, 2, 5> {
    Stencil::new(
        [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]],
        move |args: &[f64; 5]| {
            let middle = args[0];
            let left = args[1];
            let right = args[2];
            let bottom = args[3];
            let top = args[4];
            middle
                + (k_x * dt / (dx * dx)) * (left - 2.0 * middle + right)
                + (k_y * dt / (dy * dy)) * (top - 2.0 * middle + bottom)
        },
    )
}
