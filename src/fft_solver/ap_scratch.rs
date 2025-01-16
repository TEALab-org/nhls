use crate::domain::*;
use crate::fft_solver::*;
use crate::util::*;

fn blocks_to_double(block_requirement: usize) -> usize {
    (block_requirement * MIN_ALIGNMENT) / std::mem::size_of::<f64>()
}

pub struct APScratch {
    buffer: AlignedVec<f64>,
}

impl APScratch {
    pub fn allocate(root_requirement: usize) -> Self {
        APScratch {
            buffer: AlignedVec::new(blocks_to_double(root_requirement)),
        }
    }

    pub fn root_scratch<'a>(&'a mut self) -> APNodeScratch<'a> {
        APNodeScratch {
            buffer: bytemuck::cast_slice_mut(&mut self.buffer),
        }
    }
}

pub struct APNodeScratch<'a> {
    buffer: &'a mut [u8],
}

impl<'a> APNodeScratch<'a> {
    // split off scratch space
    pub fn split_scratch(mut self, blocks: usize) -> (Self, Self) {
        let bytes = blocks * MIN_ALIGNMENT;
        debug_assert!(self.buffer.len() >= bytes);
        let (scratch_buffer, remainder_buffer) =
            self.buffer.split_at_mut(bytes);
        let scratch = APNodeScratch {
            buffer: scratch_buffer,
        };
        let remainder = APNodeScratch {
            buffer: remainder_buffer,
        };

        (scratch, remainder)
    }

    pub fn split_io_domains<const GRID_DIMENSION: usize>(
        mut self,
        aabb: AABB<GRID_DIMENSION>,
    ) -> ([SliceDomain<'a, GRID_DIMENSION>; 2], APNodeScratch<'a>) {
        let min_bytes = aabb.buffer_size() * std::mem::size_of::<f64>();
        let blocks = min_bytes.div_ceil(MIN_ALIGNMENT);
        let bytes = blocks * MIN_ALIGNMENT;
        debug_assert!(self.buffer.len() >= bytes);
        let (input_buffer, remainder_buffer) = self.buffer.split_at_mut(bytes);
        let (output_buffer, remainder_buffer) =
            remainder_buffer.split_at_mut(bytes);

        let input_domain =
            SliceDomain::new(aabb, bytemuck::cast_slice_mut(input_buffer));
        let output_domain =
            SliceDomain::new(aabb, bytemuck::cast_slice_mut(output_buffer));
        let remainder = APNodeScratch {
            buffer: remainder_buffer,
        };

        ([input_domain, output_domain], remainder)
    }

    // split off complex buffer
    pub fn split_complex_buffer<const GRID_DIMENSION: usize>(
        &'a mut self,
        aabb: AABB<GRID_DIMENSION>,
    ) -> &'a mut [c64] {
        let min_bytes = aabb.complex_buffer_size() * std::mem::size_of::<c64>();
        let blocks = min_bytes.div_ceil(MIN_ALIGNMENT);
        let bytes = blocks * MIN_ALIGNMENT;
        debug_assert!(self.buffer.len() >= bytes);
        let (complex_buffer, _remainder_buffer) =
            self.buffer.split_at_mut(bytes);
        let result = bytemuck::cast_slice_mut(complex_buffer);
        result
    }

    pub fn unsafe_complex_buffer<const GRID_DIMENSION: usize>(
        &'a self,
        aabb: AABB<GRID_DIMENSION>,
    ) -> &'a mut [c64] {
        // We can use the whole slice for this, we return it right after
        let len = self.buffer.len();
        let buffer_ptr = self.buffer.as_ptr();
        let buffer_mut = unsafe {
            let buffer_ptr_mut = buffer_ptr as *mut u8;
            std::slice::from_raw_parts_mut(buffer_ptr_mut, len)
        };
        let result = bytemuck::cast_slice_mut(buffer_mut);
        debug_assert!(aabb.complex_buffer_size() <= result.len());
        result
    }
}
