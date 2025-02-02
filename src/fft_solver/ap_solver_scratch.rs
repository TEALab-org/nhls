use crate::fft_solver::*;
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
    scratch_ptr: SyncConstPtr<u8>,
    size: usize,
}

impl Drop for ScratchSpace {
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

impl ScratchSpace {
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
        ScratchSpace { scratch_ptr, size }
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
