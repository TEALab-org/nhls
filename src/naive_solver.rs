use crate::apply_iter::*;
use crate::domain;
use crate::linear_stencil;

pub struct Naive1DSolver {
    domain: domain::APBlockDomainD1,
    system: linear_stencil::System,
}

impl Naive1DSolver {
    pub fn new(domain: domain::APBlockDomainD1, system: linear_stencil::System) -> Naive1DSolver {
        // Check BC sizes

        Naive1DSolver { domain, system }
    }

    pub fn solve(&self) {
        // Create buffer
        let bc_size = self.domain.boundary_values.len();
        let buffer_width: usize = bc_size * 2 + self.domain.zone_widths.iter().sum::<usize>();
        let mut buffer_1 = vec![0.0f32; buffer_width];
        for j in 0..buffer_width - bc_size {
            let i = j - bc_size;
            let v = (i * 13 + 8) as f32;
            buffer_1[j] = v;
        }

        // Set BCs
        buffer_1[0..bc_size].copy_from_slice(&self.domain.boundary_values);
        buffer_1[buffer_width - bc_size..buffer_width]
            .copy_from_slice(&self.domain.boundary_values);

        let mut buffer_2 = buffer_1.clone();

        // Loop over time
        for _ in 0..self.domain.final_t {
            // apply stencils
            let apply_iter = ApplicationIter::new(&mut buffer_2, &self.domain.zone_widths, bc_size);
            apply_iter.for_each(|application| {
                let result: f32 = linear_stencil::apply_stencil(
                    &self.system.stencils[application.stencil],
                    &buffer_1,
                    application.index as i32,
                );
                *application.output = result;
            });

            // swap buffers
            std::mem::swap(&mut buffer_1, &mut buffer_2);
        }

        println!("{:?}", buffer_1);
    }
}
