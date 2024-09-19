use nhls::linear_stencil::*;
use nhls::domain::*;
use nhls::naive_solver::*;

fn main() {
    // Match Michael's D1 example
    let system = System {
        stencils: vec![
            StencilD1 {
                components: vec![
                    ComponentD1 {
                        offset: -2,
                        factor: 0.3,
                    },
                    ComponentD1 {
                        offset: 0,
                        factor: 0.5,
                    },
                    ComponentD1 {
                        offset: 1,
                        factor: 0.1,
                    },
                    ComponentD1 {
                        offset: 7,
                        factor: 0.1,
                    },
                ],
            },
            StencilD1 {
                components: vec![
                    ComponentD1 {
                        offset: -1,
                        factor: 0.3,
                    },
                    ComponentD1 {
                        offset: 0,
                        factor: 0.5,
                    },
                    ComponentD1 {
                        offset: 1,
                        factor: 0.2,
                    },
                ],
            },
            StencilD1 {
                components: vec![
                    ComponentD1 {
                        offset: -6,
                        factor: 0.4,
                    },
                    ComponentD1 {
                        offset: 0,
                        factor: 0.4,
                    },
                    ComponentD1 {
                        offset: 1,
                        factor: 0.2,
                    },
                ],
            },
        ],
    };

    let domain = APBlockDomainD1 {
        boundary_values: vec![1.0; 10],
        zone_widths: vec![10000, 10000, 15000],
        final_t: 40000,
    };

    let solver = Naive1DSolver::new(domain, system);
    solver.solve();
}
