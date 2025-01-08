use crate::domain::*;
use crate::util::*;

/// Dynamically create and drop SliceDomains from single allocaiton
pub struct DomainStack<'a, const GRID_DIMENSION: usize> {
    buffer: AlignedVec<f64>,
    remainder: &'a mut [f64],
    next_id: usize,
}

impl<'a, const GRID_DIMENSION: usize> DomainStack<'a, GRID_DIMENSION> {
    pub fn buffer(&'a self) -> &'a [f64] {
        self.buffer.as_slice()
    }

    pub fn remainder(&'a self) -> &'a [f64] {
        self.remainder
    }

    pub fn with_capacity(size: usize) -> Self {
        // create remainder from buffer parts
        let buffer = AlignedVec::new(size);
        let data_ptr = buffer.as_slice().as_ptr();
        let next_id = 0;
        let remainder = unsafe {
            let data_ptr_mut = data_ptr as *mut f64;
            std::slice::from_raw_parts_mut(data_ptr_mut, size)
        };

        DomainStack {
            remainder,
            buffer,
            next_id,
        }
    }

    pub fn pop_domain(
        &mut self,
        aabb: AABB<GRID_DIMENSION>,
    ) -> (DomainId, SliceDomain<'a, GRID_DIMENSION>) {
        let buffer_size = aabb.buffer_size();
        let remainder_len = self.remainder.len();
        debug_assert!(remainder_len >= buffer_size);
        let remainder_ptr = self.remainder.as_ptr();

        let new_remainder_len = remainder_len - buffer_size;

        let slice = unsafe {
            let slice_ptr = remainder_ptr.add(new_remainder_len);
            let slice_ptr_mut = slice_ptr as *mut f64;
            std::slice::from_raw_parts_mut(slice_ptr_mut, buffer_size)
        };

        self.remainder = unsafe {
            let remainder_ptr_mut = remainder_ptr as *mut f64;
            std::slice::from_raw_parts_mut(remainder_ptr_mut, new_remainder_len)
        };

        let result = (self.next_id, SliceDomain::new(aabb, slice));
        self.next_id += 1;
        result
    }

    pub fn push_domain(
        &mut self,
        id: DomainId,
        domain: SliceDomain<'a, GRID_DIMENSION>,
    ) {
        debug_assert!(self.next_id != 0);
        debug_assert_eq!(self.next_id - 1, id);
        self.next_id -= 1;
        let buffer_size = domain.aabb().buffer_size();
        let remainder_size = self.remainder.len();
        self.remainder = unsafe {
            let remainder_ptr = self.remainder.as_ptr();
            let remainder_ptr_mut = remainder_ptr as *mut f64;
            std::slice::from_raw_parts_mut(
                remainder_ptr_mut,
                buffer_size + remainder_size,
            )
        };
    }
}

#[cfg(test)]
mod unit_test {
    use super::*;

    #[test]
    fn counter() {
        let size = 100;
        let mut counter = DomainStack::with_capacity(size);
        let aabb_1 = AABB::new(matrix![0, 1]);
        let aabb_1_s = aabb_1.buffer_size();
        let (id_1, domain_1) = counter.pop_domain(aabb_1);
        debug_assert_eq!(id_1, 0);

        debug_assert_eq!(counter.buffer().len(), size);
        debug_assert_eq!(domain_1.buffer().len(), aabb_1_s);
        debug_assert_eq!(aabb_1_s + counter.remainder().len(), size);
        unsafe {
            let expected_domain_1_ptr =
                counter.buffer().as_ptr().add(size - aabb_1_s);
            debug_assert_eq!(domain_1.buffer().as_ptr(), expected_domain_1_ptr);
        }

        let aabb_2 = AABB::new(matrix![0, 3]);
        let aabb_2_s = aabb_2.buffer_size();
        let (id_2, domain_2) = counter.pop_domain(aabb_2);
        debug_assert_eq!(id_2, 1);
        debug_assert_eq!(counter.buffer().len(), size);
        debug_assert_eq!(domain_2.buffer().len(), aabb_2_s);
        debug_assert_eq!(aabb_2_s + aabb_1_s + counter.remainder().len(), size);
        unsafe {
            let expected_domain_2_ptr =
                counter.buffer().as_ptr().add(size - (aabb_1_s + aabb_2_s));
            debug_assert_eq!(domain_2.buffer().as_ptr(), expected_domain_2_ptr);
        }

        counter.push_domain(id_2, domain_2);

        let aabb_3 = AABB::new(matrix![0, 5]);
        let aabb_3_s = aabb_3.buffer_size();
        let (id_3, domain_3) = counter.pop_domain(aabb_3);
        debug_assert_eq!(id_3, 1);
        debug_assert_eq!(counter.buffer().len(), size);
        debug_assert_eq!(domain_3.buffer().len(), aabb_3_s);
        debug_assert_eq!(aabb_3_s + aabb_1_s + counter.remainder().len(), size);
        unsafe {
            let expected_domain_3_ptr =
                counter.buffer().as_ptr().add(size - (aabb_1_s + aabb_3_s));
            debug_assert_eq!(domain_3.buffer().as_ptr(), expected_domain_3_ptr);
        }
    }
}
