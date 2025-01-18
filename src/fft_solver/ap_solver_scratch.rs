use crate::fft_solver::*;
use crate::util::*;
use std::mem::size_of;

pub type AllocationType = f64;

pub struct ScratchDescriptor {
    input_offset: usize,
    output_offset: usize,
    buffer_size: usize,
    complex_offset: usize,
    complex_size: usize,
    size: usize,
}

pub struct ScratchSpace {
    //scratch_descriptors: Vec<ScratchDescriptor>,
    scratch_allocation: AlignedVec<AllocationType>,
    scratch_ptr: *const u8,
}

impl ScratchSpace {
    pub fn new(bytes: usize) -> Self {
        let scratch_allocations_len = bytes / size_of::<AllocationType>();
        let scratch_allocation = AlignedVec::new(scratch_allocations_len);
        let scratch_ptr = scratch_allocation.as_slice().as_ptr() as *const u8;
        debug_assert!(scratch_ptr as usize / MIN_ALIGNMENT == 0);
        ScratchSpace {
            scratch_allocation,
            scratch_ptr,
        }
    }

    /// offset in bytes
    /// len in bytes
    pub fn unsafe_get_buffer<
        T: bytemuck::NoUninit + bytemuck::AnyBitPattern,
    >(
        &self,
        offset: usize,
        len: usize,
    ) -> &mut [T] {
        let scratch_bytes = unsafe {
            let scratch_ptr_mut = self.scratch_ptr.add(offset) as *mut u8;
            std::slice::from_raw_parts_mut(scratch_ptr_mut, len)
        };
        bytemuck::cast_slice_mut(scratch_bytes)
    }
}
