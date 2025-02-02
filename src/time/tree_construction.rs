use std::collections::HashMap;

type Stencil = HashMap<(i32, i32, i32), f64>; // Representing the stencil as a map of multi-dimensional offsets to values (constant + time_func)
type StencilTree = HashMap<(i32, i32), Stencil>; // Tree mapping (start, end) range to a stencil

#[derive(Debug, Clone)]
struct GridDimensions {
    dims: Vec<i32>, // Store grid sizes for each dimension
}

impl GridDimensions {
    fn new(dims: Vec<i32>) -> Self {
        GridDimensions { dims }
    }

    // Normalize an offset for a specific dimension
    fn normalize_offset(&self, offset: i32, dim: usize) -> i32 {
        let size = self.dims[dim] as i32;
        offset % size
    }
}

fn build_stencil_tree(s: &HashMap<(i32, i32, i32), (f64, Option<Box<dyn Fn(i32) -> f64>>)>, t: i32, grid_dims: &GridDimensions) -> StencilTree {
    let mut tree = StencilTree::new();

    // Compute the stencil at a specific time
    fn compute_stencil_at_time(s: &HashMap<(i32, i32, i32), (f64, Option<Box<dyn Fn(i32) -> f64>>)>, t: i32) -> Stencil {
        let mut stencil = Stencil::new();
        for (&offset, &(constant, ref time_func)) in s.iter() {
            let value = constant + time_func.as_ref().map_or(0.0, |func| func(t));
            stencil.insert(offset, value);
        }
        stencil
    }

    // Normalize the stencil to ensure offsets are within the grid bounds
    fn normalize_stencil(tree: &mut StencilTree, stencil_key: &(i32, i32), grid_dims: &GridDimensions) {
        if let Some(stencil) = tree.get(stencil_key) {
            let mut normalized = Stencil::new();
            for (&offset, &value) in stencil.iter() {
                let mut normalized_offset = vec![];
                for (i, dim) in offset.iter().enumerate() {
                    let wrapped_offset = grid_dims.normalize_offset(*dim, i);
                    normalized_offset.push(wrapped_offset);
                }
                let normalized_key = (normalized_offset[0], normalized_offset[1], normalized_offset[2]);
                let entry = normalized.entry(normalized_key).or_insert(0.0);
                *entry += value;
            }
            tree.insert(*stencil_key, normalized);
        }
    }

    // Combine two stencils
    fn combine_stencils(left: &Stencil, right: &Stencil) -> Stencil {
        let mut combined = left.clone();
        for (&offset, &value) in right.iter() {
            let entry = combined.entry(offset).or_insert(0.0);
            *entry += value;
        }
        combined
    }

    // Recursively build the stencil tree
    fn build_range(start: i32, end: i32, s: &HashMap<(i32, i32, i32), (f64, Option<Box<dyn Fn(i32) -> f64>>)>, grid_dims: &GridDimensions, tree: &mut StencilTree) -> Stencil {
        if start == end {
            let stencil = compute_stencil_at_time(s, start);
            tree.insert((start, end), stencil.clone());
            normalize_stencil(tree, &(start, end), grid_dims);
            stencil
        } else {
            let mid = (start + end) / 2;
            let left_stencil = build_range(start, mid, s, grid_dims, tree);
            let right_stencil = build_range(mid + 1, end, s, grid_dims, tree);
            let combined_stencil = combine_stencils(&left_stencil, &right_stencil);
            tree.insert((start, end), combined_stencil.clone());
            normalize_stencil(tree, &(start, end), grid_dims);
            combined_stencil
        }
    }

    // Start building the tree from time 1 to t
    build_range(1, t, s, grid_dims, &mut tree);

    tree
}

fn main() {
    // Example usage: creating a grid of 3 dimensions
    let grid_dims = GridDimensions::new(vec![10, 10, 10]);

    let mut stencil_map = HashMap::new();
    stencil_map.insert((-1, 1, 0), (1.0, None));
    stencil_map.insert((0, 1, 0), (1.0, Some(Box::new(|t| t as f64))));
    stencil_map.insert((1, 1, 0), (2.0, None));

    let tree = build_stencil_tree(&stencil_map, 5, &grid_dims);
    for ((start, end), stencil) in &tree {
        println!("Stencil for range ({}, {}): {:?}", start, end, stencil);
    }
}
