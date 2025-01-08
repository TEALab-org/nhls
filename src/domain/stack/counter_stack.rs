use crate::domain::*;
use crate::util::*;

/// Mimics the domain stack during runtime,
/// used to compute the size of the runtime stack.
pub struct CounterStack<const GRID_DIMENSION: usize> {
    next_id: usize,
    size: usize,
    max_size: usize,
}

impl<const GRID_DIMENSION: usize> CounterStack<GRID_DIMENSION> {
    pub fn blank() -> Self {
        CounterStack {
            next_id: 0,
            size: 0,
            max_size: 0,
        }
    }

    pub fn pop_domain(&mut self, aabb: &AABB<GRID_DIMENSION>) -> DomainId {
        let buffer_size = aabb.buffer_size();
        let result = self.next_id;
        self.next_id += 1;
        self.size += buffer_size;
        if self.size > self.max_size {
            self.max_size = self.size;
        }
        result
    }

    pub fn push_domain(&mut self, id: DomainId, aabb: &AABB<GRID_DIMENSION>) {
        let buffer_size = aabb.buffer_size();
        debug_assert!(self.next_id != 0);
        debug_assert!(self.size >= buffer_size);
        debug_assert_eq!(self.next_id - 1, id);
        self.size -= buffer_size;
        self.next_id -= 1;
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn max_size(&self) -> usize {
        self.max_size
    }

    pub fn finish(self) -> usize {
        self.max_size
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn counter() {
        let mut counter = CounterStack::blank();
        let aabb_1 = AABB::new(matrix![0, 1]);
        let aabb_1_s = aabb_1.buffer_size();
        debug_assert_eq!(counter.size(), 0);
        debug_assert_eq!(counter.max_size(), 0);
        let id_1 = counter.pop_domain(&aabb_1);
        debug_assert_eq!(id_1, 0);
        debug_assert_eq!(counter.size(), aabb_1_s);
        debug_assert_eq!(counter.max_size(), aabb_1_s);

        let aabb_2 = AABB::new(matrix![0, 3]);
        let aabb_2_s = aabb_2.buffer_size();
        let id_2 = counter.pop_domain(&aabb_2);
        debug_assert_eq!(id_2, 1);
        debug_assert_eq!(counter.max_size(), aabb_1_s + aabb_2_s);

        counter.push_domain(id_2, &aabb_2);
        debug_assert_eq!(counter.size(), aabb_1_s);
        debug_assert_eq!(counter.max_size(), aabb_1_s + aabb_2_s);

        let aabb_3 = AABB::new(matrix![0, 5]);
        let aabb_3_s = aabb_3.buffer_size();
        let id_3 = counter.pop_domain(&aabb_3);
        debug_assert_eq!(id_3, 1);
        debug_assert_eq!(counter.size(), aabb_1_s + aabb_3_s);
        debug_assert_eq!(counter.max_size(), aabb_1_s + aabb_3_s);
    }
}
