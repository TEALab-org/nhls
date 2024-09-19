#[derive(Debug)]
pub struct StencilApplication<'a> {
    pub output: &'a mut f32,
    pub index: usize,
    pub stencil: usize,
}

#[derive(Debug)]
pub struct ApplicationIter<'a> {
    // Remaini
    output_buffer: &'a mut [f32],

    // Which zone are we in
    zone_index: usize,

    // What is the current output index
    current_index: usize,

    // When do we switch zones?
    next_zone: usize,

    // What are the zone widths
    zone_widths: &'a [usize],

    // Are we done?
    done: bool,
}

impl<'a> ApplicationIter<'a> {
    pub fn new(
        output_buffer: &'a mut [f32],
        zone_widths: &'a [usize],
        start_offset: usize,
    ) -> ApplicationIter<'a> {
        let (_, remaining) = output_buffer.split_at_mut(start_offset);
        ApplicationIter {
            output_buffer: remaining,
            zone_index: 0,
            current_index: start_offset,
            next_zone: start_offset + zone_widths[0],
            zone_widths,
            done: false,
        }
    }
}

impl<'a> std::iter::Iterator for ApplicationIter<'a> {
    type Item = StencilApplication<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if current index is within bounds
        if self.done {
            return None;
        }

        // Create Stencil Application
        // Note replace trick is from
        // https://users.rust-lang.org/t/how-does-vecs-iterator-return-a-mutable-reference/60235/7
        let output_buffer = std::mem::take(&mut self.output_buffer);
        let (output, remaining) = output_buffer.split_at_mut(1);
        self.output_buffer = remaining;
        let result = StencilApplication {
            output: &mut output[0],
            index: self.current_index,
            stencil: self.zone_index,
        };

        // Update indices
        self.current_index += 1;
        if self.current_index >= self.next_zone {
            if self.zone_index + 1 == self.zone_widths.len() {
                self.done = true;
            } else {
                self.zone_index += 1;
                self.next_zone += self.zone_widths[self.zone_index];
            }
        }

        Some(result)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn iter_test_1() {
        let mut output_buffer = Vec::with_capacity(10);
        for i in 0..10 {
            output_buffer.push(i as f32);
        }
        let zone_widths = vec![2, 3, 3];
        let start_offset = 1;

        let iter = ApplicationIter::new(&mut output_buffer, &zone_widths, start_offset);
        let apps: Vec<StencilApplication> = iter.collect();

        assert_eq!(apps.len(), 8);
        let stencils = [0, 0, 1, 1, 1, 2, 2, 2];
        for i in 0..8 {
            assert_eq!(*apps[i].output as usize, i + 1);
            assert_eq!(apps[i].index, i + 1);
            assert_eq!(apps[i].stencil, stencils[i]);
        }
    }
}
