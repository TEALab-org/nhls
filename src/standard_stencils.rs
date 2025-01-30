use crate::stencil::*;

// Heat stencils in 1D, 2D, and 3D

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

pub fn heat_3d(
    dt: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    k_x: f64,
    k_y: f64,
    k_z: f64,
) -> StencilF64<impl StencilOperation<f64, 7>, 3, 7> {
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

// Jacobi (Average) stencisl in 1D, 2D, and 3D

pub fn jacobi_1d(
    weight: f64,
) -> StencilF64<impl StencilOperation<f64, 3>, 1, 3> {
    Stencil::new([[-1], [0], [1]], move |args: &[f64; 3]| {
        let center = args[1];
        let weighted_sum = weight * (args[0] + args[2]);
        weighted_sum + (1.0 - weight) * center
    })
}

pub fn jacobi_2d(
    weight: f64,
) -> StencilF64<impl StencilOperation<f64, 9>, 2, 9> {
    Stencil::new(
        [
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 0],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1],
        ],
        move |args: &[f64; 9]| {
            let center = args[4];
            let weighted_sum = weight
                * (args[0]
                    + args[1]
                    + args[2]
                    + args[3]
                    + args[5]
                    + args[6]
                    + args[7]
                    + args[8]);
            weighted_sum + (1.0 - weight) * center
        },
    )
}

pub fn jacobi_3d(
    weight: f64,
) -> StencilF64<impl StencilOperation<f64, 27>, 3, 27> {
    Stencil::new(
        [
            [-1, -1, -1],
            [-1, -1, 0],
            [-1, -1, 1],
            [-1, 0, -1],
            [-1, 0, 0],
            [-1, 0, 1],
            [-1, 1, -1],
            [-1, 1, 0],
            [-1, 1, 1],
            [0, -1, -1],
            [0, -1, 0],
            [0, -1, 1],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, -1],
            [0, 1, 0],
            [0, 1, 1],
            [1, -1, -1],
            [1, -1, 0],
            [1, -1, 1],
            [1, 0, -1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, -1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        move |args: &[f64; 27]| {
            let center = args[13];
            let weighted_sum = weight
                * args
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != 13)
                    .map(|(_, &val)| val)
                    .sum::<f64>();
            weighted_sum + (1.0 - weight) * center
        },
    )
}

// Guassian Blur stencils in 1D, 2D, and 3D

pub fn gaussian_blur_1d() -> StencilF64<impl StencilOperation<f64, 3>, 1, 3> {
    Stencil::new([[-1], [0], [1]], move |args: &[f64; 3]| {
        let kernel = [1.0, 2.0, 1.0];
        let weighted_sum: f64 =
            args.iter().zip(kernel.iter()).map(|(a, k)| a * k).sum();
        weighted_sum / 4.0
    })
}

pub fn gaussian_blur_2d() -> StencilF64<impl StencilOperation<f64, 9>, 2, 9> {
    Stencil::new(
        [
            [-1, -1],
            [0, -1],
            [1, -1],
            [-1, 0],
            [0, 0],
            [1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
        ],
        move |args: &[f64; 9]| {
            let kernel = [1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
            let weighted_sum: f64 =
                args.iter().zip(kernel.iter()).map(|(a, k)| a * k).sum();
            weighted_sum / 16.0
        },
    )
}

pub fn gaussian_blur_3d() -> StencilF64<impl StencilOperation<f64, 27>, 3, 27> {
    Stencil::new(
        [
            [-1, -1, -1],
            [-1, -1, 0],
            [-1, -1, 1],
            [-1, 0, -1],
            [-1, 0, 0],
            [-1, 0, 1],
            [-1, 1, -1],
            [-1, 1, 0],
            [-1, 1, 1],
            [0, -1, -1],
            [0, -1, 0],
            [0, -1, 1],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, -1],
            [0, 1, 0],
            [0, 1, 1],
            [1, -1, -1],
            [1, -1, 0],
            [1, -1, 1],
            [1, 0, -1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, -1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        move |args: &[f64; 27]| {
            let kernel = [
                1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0, 2.0, 4.0, 2.0,
                4.0, 8.0, 4.0, 2.0, 4.0, 8.0, 4.0, 1.0, 2.0, 1.0, 2.0, 4.0,
                2.0, 1.0, 2.0, 1.0,
            ];
            let weighted_sum: f64 =
                args.iter().zip(kernel.iter()).map(|(a, k)| a * k).sum();
            weighted_sum / 64.0
        },
    )
}

// Linear possion (discrete Laplacian operator) stencils in 1D, 2D, and 3D

pub fn poisson_1d() -> StencilF64<impl StencilOperation<f64, 3>, 1, 3> {
    Stencil::new([[-1], [0], [1]], move |args: &[f64; 3]| {
        let left = args[0];
        let center = args[1];
        let right = args[2];
        -2.0 * center + left + right
    })
}

pub fn poisson_2d() -> StencilF64<impl StencilOperation<f64, 5>, 2, 5> {
    Stencil::new(
        [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]],
        move |args: &[f64; 5]| {
            let left = args[0];
            let right = args[1];
            let bottom = args[2];
            let top = args[3];
            let center = args[4];
            -4.0 * center + left + right + bottom + top
        },
    )
}

pub fn poisson_3d() -> StencilF64<impl StencilOperation<f64, 7>, 3, 7> {
    Stencil::new(
        [
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
            [0, 0, 0],
        ],
        move |args: &[f64; 7]| {
            let left = args[0];
            let right = args[1];
            let bottom = args[2];
            let top = args[3];
            let front = args[4];
            let back = args[5];
            let center = args[6];
            -6.0 * center + left + right + bottom + top + front + back
        },
    )
}
