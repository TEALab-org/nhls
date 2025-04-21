use crate::util::*;
use std::collections::HashMap;
use std::io::prelude::*;

pub type TVOpId = usize;

/// Describes a periodic solve,
/// including time-varying
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PeriodicOpDescriptor<const GRID_DIMENSION: usize> {
    pub step_min: usize,
    pub step_max: usize,
    pub exclusive_bounds: Coord<GRID_DIMENSION>,
    pub threads: usize,
}

impl<const GRID_DIMENSION: usize> PeriodicOpDescriptor<GRID_DIMENSION> {
    pub fn blank() -> Self {
        PeriodicOpDescriptor {
            step_min: 0,
            step_max: 0,
            exclusive_bounds: Coord::zeros(),
            threads: 0,
        }
    }
}

/// Collect all periodic solves needed
/// during plan creation
pub struct PeriodicOpCollector<const GRID_DIMENSION: usize> {
    pub descriptor_map: HashMap<PeriodicOpDescriptor<GRID_DIMENSION>, TVOpId>,
    next_id: usize,
}

impl<const GRID_DIMENSION: usize> PeriodicOpCollector<GRID_DIMENSION> {
    pub fn blank() -> Self {
        PeriodicOpCollector {
            descriptor_map: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn get_op_id(
        &mut self,
        descriptor: PeriodicOpDescriptor<GRID_DIMENSION>,
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

    pub fn finish(self) -> Vec<PeriodicOpDescriptor<GRID_DIMENSION>> {
        let mut result =
            vec![PeriodicOpDescriptor::blank(); self.descriptor_map.len()];
        // Collect
        for (descriptor, id) in self.descriptor_map {
            result[id] = descriptor;
        }
        result
    }

    pub fn write_query_file<P: AsRef<std::path::Path>>(&self, path: &P) {
        println!("Writing query file: {:?}", path.as_ref());
        let mut writer =
            std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        writeln!(writer, "QUERY PLAN: {}", self.descriptor_map.len()).unwrap();
        for (i, (key, value)) in self.descriptor_map.iter().enumerate() {
            writeln!(writer, "i: {}, {:?} -> {}", i, key, value).unwrap();
        }
    }
}
