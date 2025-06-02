use crate::ap_solver::index_types::*;
use crate::ap_solver::periodic_ops::*;
use crate::ap_solver::tv_periodic_ops::*;
use crate::ap_solver::tv_periodic_ops_builder::*;
use crate::stencil::TVStencil;
use crate::util::*;
use std::collections::HashMap;

use crate::ap_solver::planner::PlannerParameters;

/// Collect all periodic solves needed
/// during plan creation
pub struct TvPeriodicOpsCollector<
    'a,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
> {
    pub descriptor_map: HashMap<PeriodicOpDescriptor<GRID_DIMENSION>, OpId>,
    next_id: usize,
    stencil: &'a StencilType,
    aabb: AABB<GRID_DIMENSION>,
    params: &'a PlannerParameters<GRID_DIMENSION>,
    steps: usize,
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    >
    TvPeriodicOpsCollector<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>
{
    pub fn new(
        stencil: &'a StencilType,
        params: &'a PlannerParameters<GRID_DIMENSION>,
    ) -> Self {
        TvPeriodicOpsCollector {
            descriptor_map: HashMap::new(),
            next_id: 0,
            stencil,
            aabb: params.aabb,
            params,
            steps: params.steps,
        }
    }

    pub fn get_op_id(
        &mut self,
        descriptor: PeriodicOpDescriptor<GRID_DIMENSION>,
    ) -> OpId {
        if let Some(id) = self.descriptor_map.get(&descriptor) {
            *id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.descriptor_map.insert(descriptor, id);
            id
        }
    }

    pub fn finish(
        self,
    ) -> TvPeriodicOps<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType> {
        let mut result =
            vec![PeriodicOpDescriptor::blank(); self.descriptor_map.len()];
        // Collect
        for (descriptor, id) in self.descriptor_map {
            result[id] = descriptor;
        }

        let ops_builder = TvPeriodicOpsBuilder::new(self.stencil, self.aabb);
        ops_builder.build_op_calc(
            self.steps,
            self.params.threads,
            self.params.plan_type,
            &result,
        )
    }
}

impl<
        'a,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
        StencilType: TVStencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    >
    PeriodicOpsBuilder<
        GRID_DIMENSION,
        TvPeriodicOps<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType>,
    >
    for TvPeriodicOpsCollector<
        'a,
        GRID_DIMENSION,
        NEIGHBORHOOD_SIZE,
        StencilType,
    >
{
    fn get_op_id(
        &mut self,
        descriptor: PeriodicOpDescriptor<GRID_DIMENSION>,
    ) -> OpId {
        self.get_op_id(descriptor)
    }

    fn finish(
        self,
    ) -> TvPeriodicOps<'a, GRID_DIMENSION, NEIGHBORHOOD_SIZE, StencilType> {
        self.finish()
    }
}
