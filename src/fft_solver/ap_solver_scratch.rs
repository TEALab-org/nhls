use crate::fft_solver::*;
use crate::util::*;
use std::mem::size_of;
use sync_ptr::SyncConstPtr;

pub type AllocationType = f64;

#[derive(Copy, Clone, Debug, Default)]
pub struct ScratchDescriptor {
    pub input_offset: usize,
    pub output_offset: usize,
    pub real_buffer_size: usize,
    pub complex_offset: usize,
    pub complex_buffer_size: usize,
}

pub struct ScratchSpace {
    //scratch_descriptors: Vec<ScratchDescriptor>,
    _scratch_allocation: AlignedVec<AllocationType>,
    scratch_ptr: SyncConstPtr<u8>,
}

impl ScratchSpace {
    pub fn new(bytes: usize) -> Self {
        let scratch_allocations_len = bytes / size_of::<AllocationType>();
        let scratch_allocation = AlignedVec::new(scratch_allocations_len);
        let scratch_ptr = unsafe {
            SyncConstPtr::new(
                scratch_allocation.as_slice().as_ptr() as *const u8
            )
        };
        debug_assert!(scratch_ptr.inner() as usize % MIN_ALIGNMENT == 0);
        ScratchSpace {
            _scratch_allocation: scratch_allocation,
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
        debug_assert!(len > 0);
        let scratch_bytes = unsafe {
            let scratch_ptr_mut = self.scratch_ptr.add(offset) as *mut u8;
            debug_assert!(scratch_ptr_mut as usize % MIN_ALIGNMENT == 0);
            std::slice::from_raw_parts_mut(scratch_ptr_mut, len)
        };
        bytemuck::cast_slice_mut(scratch_bytes)
    }
}
