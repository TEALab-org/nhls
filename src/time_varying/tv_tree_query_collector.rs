use crate::util::*;
use std::collections::HashMap;

pub type TVOpId = usize;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TVOpDescriptor<const GRID_DIMENSION: usize> {
    pub step_min: usize,
    pub step_max: usize,
    pub exclusive_bounds: Coord<GRID_DIMENSION>,
}

pub struct TVTreeQueryCollector<const GRID_DIMENSION: usize> {
    descriptor_map: HashMap<TVOpDescriptor<GRID_DIMENSION>, TVOpId>,
    next_id: usize,
}

impl<const GRID_DIMENSION: usize> TVTreeQueryCollector<GRID_DIMENSION> {
    pub fn new() -> Self {
        TVTreeQueryCollector {
            descriptor_map: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn get_op_id(
        &mut self,
        descriptor: TVOpDescriptor<GRID_DIMENSION>,
    ) -> TVOpId {
        if let Some(id) = self.descriptor_map.get(&descriptor) {
            *id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.descriptor_map.insert(descriptor, id);
            id
        }
    }
}
