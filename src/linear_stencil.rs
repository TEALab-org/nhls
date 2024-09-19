pub struct ComponentD1 {
    pub offset: i32,
    pub factor: f32,
}

pub struct StencilD1 {
    pub components: Vec<ComponentD1>,
}

pub struct System {
    pub stencils: Vec<StencilD1>,
}

pub fn apply_stencil(stencil: &StencilD1, buffer: &[f32], i: i32) -> f32 {
    let mut sum = 0.0;
    for component in &stencil.components {
        let n_i: usize = (i + component.offset) as usize;
        sum += buffer[n_i] * component.factor;
    }
    sum
}
