use std::collections::HashMap;

// Define a type for the time-varying weight function
type TimeVaryingWeightFn = Box<dyn Fn(f64) -> f64>;

// Struct to represent Michael's hashmap based stencil
struct Stencil {
    // Mapping of d-dimensional coordinates (as Vec<i32>) to (constant weight, time-varying weight function)
    pub stencil_map: HashMap<Vec<i32>, (f64, Option<TimeVaryingWeightFn>)>,
}

impl Stencil {
    // Constructor to create a new stencil
    pub fn new() -> Self {
        Stencil {
            stencil_map: HashMap::new(),
        }
    }

    // Method to insert a stencil entry with constant and time-varying weight
    pub fn insert(
        &mut self,
        coords: Vec<i32>,
        constant: f64,
        time_varying_weight_fn: Option<TimeVaryingWeightFn>,
    ) {
        self.stencil_map.insert(coords, (constant, time_varying_weight_fn));
    }

    // Method to get the value for a specific coordinate (with constant and/or time-varying weight)
    pub fn get_weight_at(&self, coords: &Vec<i32>, time: f64) -> f64 {
        if let Some((constant, Some(time_varying_fn))) = self.stencil_map.get(coords) {
            constant + time_varying_fn(time) // Apply time-varying function
        } else if let Some((constant, None)) = self.stencil_map.get(coords) {
            *constant // Return constant if no time-varying weight
        } else {
            0.0 // Return 0 if the coordinate is not found (default case)
        }
    }
}

//Some example time varying functions

fn constant_t_function(x: f64) -> TimeVaryingWeightFn {
    Box::new(move |_time| {
        x
    })
}

fn linear_t_function(x: f64) -> TimeVaryingWeightFn {
    Box::new(move |time| {
        x * time
    })
}

fn quadratic_t_function(x: f64) -> TimeVaryingWeightFn {
    Box::new(move |time| {
       x * time * time
    })
}


// Some manual tests to make sure the get_weight_at function works
fn main() {

    let mut stencil = Stencil::new();

    // Insert stencil entries
    stencil.insert(
        vec![-1], 
        0.5,
        Some(constant_t_function(0.1)),
    );
    
    stencil.insert(
        vec![0],
        1.0,
        Some(linear_t_function(0.1)),
    );

    stencil.insert(
        vec![1],
        1.0,
        Some(quadratic_t_function(0.1)),          
    );

    stencil.insert(
        vec![2],
        3.0,
        None,          
    );

    println!("Weight at [-1] with t=2: {}", stencil.get_weight_at(&vec![-1], 2.0));
    println!("Weight at [0] with t=2: {}", stencil.get_weight_at(&vec![0], 2.0));
    println!("Weight at [1] with t=2: {}", stencil.get_weight_at(&vec![1], 2.0));
    println!("Weight at [2] with t=2: {}", stencil.get_weight_at(&vec![2], 2.0));
    
}
