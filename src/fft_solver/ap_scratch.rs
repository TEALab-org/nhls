use crate::fft_solver::*;
use sync_ptr::SyncConstPtr;

pub type AllocationType = f64;

/// Note all nodes will need all of these
/// but we will set any values that are needed.
#[derive(Copy, Clone, Debug, Default)]
pub struct ScratchDescriptor {
    /// Offset for input domain
    pub input_offset: usize,

    /// Offset for output domain
    pub output_offset: usize,

    /// Size (in bytes) for input / output domains
    pub real_buffer_size: usize,

    /// Offset for complex buffer
    pub complex_offset: usize,

    /// Size (in bytes) for complex buffer
    pub complex_buffer_size: usize,
}

pub struct APScratch {
    scratch_ptr: SyncConstPtr<u8>,
    pub size: usize,
}

impl Drop for APScratch {
    fn drop(&mut self) {
        let alloc_layout =
            std::alloc::Layout::from_size_align(self.size, MIN_ALIGNMENT)
                .unwrap();

        unsafe {
            std::alloc::dealloc(
                self.scratch_ptr.inner() as *mut u8,
                alloc_layout,
            );
        }
    }
}

impl APScratch {
    pub fn new(size: usize) -> Self {
        let alloc_layout =
            std::alloc::Layout::from_size_align(size, MIN_ALIGNMENT).unwrap();
        let scratch_ptr = unsafe {
            SyncConstPtr::new(std::alloc::alloc(alloc_layout) as *const u8)
        };
        debug_assert!(
            scratch_ptr.inner() as usize % MIN_ALIGNMENT == 0,
            "ERROR: scratch_ptr: {}, mod: {}",
            scratch_ptr.inner() as usize,
            scratch_ptr.inner() as usize % MIN_ALIGNMENT
        );
        APScratch { scratch_ptr, size }
    }

    /// Only use this function with the values provided by APScratchBuilder
    /// This is very unsafe!
    /// offset in bytes
    /// len in bytes
    pub fn unsafe_get_buffer<
        'a,
        'b,
        T: bytemuck::NoUninit + bytemuck::AnyBitPattern,
    >(
        &'a self,
        offset: usize,
        len: usize,
    ) -> &'b mut [T] {
        debug_assert!(len > 0);
        let scratch_bytes = unsafe {
            let scratch_ptr_mut = self.scratch_ptr.add(offset) as *mut u8;
            debug_assert!(scratch_ptr_mut as usize % MIN_ALIGNMENT == 0);
            std::slice::from_raw_parts_mut(scratch_ptr_mut, len)
        };
        bytemuck::cast_slice_mut(scratch_bytes)
    }
}
