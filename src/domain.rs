/// A-periodic Block domain in one dimension
pub struct APBlockDomainD1 {
    pub boundary_values: Vec<f32>,
    pub zone_widths: Vec<usize>,
    pub final_t: usize,
}
