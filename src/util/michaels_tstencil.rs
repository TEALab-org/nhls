use std::collections::HashMap;
use std::f64::EPSILON;



///Quadratic stencil mulitplication
pub fn naive_mult(s1: &HashMap<Vec<i32>, f64>, s2: &HashMap<Vec<i32>, f64>) -> HashMap<Vec<i32>, f64>{
    let mut combined = HashMap::new();

    // Iterate over the first stencil
    for (offset1, value1) in s1 {
        // Iterate over the second stencil
        for (offset2, value2) in s2 {
            // Combine offsets via addition (like adding polynomials)
            let new_offset: Vec<i32> = offset1.iter().zip(offset2.iter()).map(|(a, b)| a + b).collect();
            
            // Multiply the values and accumulate them in the new offset
            let combined_value = value1 * value2;
            *combined.entry(new_offset).or_insert(0.0) += combined_value;
        }
    }

    combined
}

/*
///n^1.5 stencil mulitplication
pub fn karatsuba(s1: HashMap<Vec<i32>, f64>, s2: HashMap<Vec<i32>, f64>){

}
*/

pub fn fftn(input: &mut Vec<f64>) -> Vec<f64> {
    input.clone() // Replace this with actual FFT logic
}

pub fn ifftn(input: &mut Vec<f64>) -> Vec<f64> {
    input.clone() // Replace this with actual IFFT logic
}

///nlogn stencil mulitplication
pub fn fft_mult(
    s1: HashMap<Vec<i32>, f64>,
    s2: HashMap<Vec<i32>, f64>,
) -> HashMap<i32, f64> {
    // Step 1: Calculate the size of the result based on stencil lengths
    let len1 = s1.keys().map(|k| k[0]).max().unwrap() - s1.keys().map(|k| k[0]).min().unwrap() + 1;
    let len2 = s2.keys().map(|k| k[0]).max().unwrap() - s2.keys().map(|k| k[0]).min().unwrap() + 1;

    let result_len = len1 + len2 - 1;

    // Step 2: Prepare the padded stencil arrays
    // Convert result_len to usize before using it to initialize a vector
    let result_len_usize = result_len as usize;

    let mut padded_stencil1 = vec![0.0; result_len_usize];
    let mut padded_stencil2 = vec![0.0; result_len_usize];

    // Fill the padded arrays with values from the stencils
    for (offset, value) in &s1 {
        let index = (offset[0] + (result_len as i32) / 2) as usize; // Align with center
        padded_stencil1[index] = *value;
    }

    for (offset, value) in &s2 {
        let index = (offset[0] + (result_len as i32) / 2) as usize; // Align with center
        padded_stencil2[index] = *value;
    }

    // Step 3: Perform FFT on both stencils
    let fft_s1 = fftn(&mut padded_stencil1);
    let fft_s2 = fftn(&mut padded_stencil2);

    // Step 4: Perform pointwise multiplication
    let mut multiplied_result = vec![0.0; result_len_usize];
    for i in 0..result_len_usize {
        multiplied_result[i] = fft_s1[i] * fft_s2[i];
    }

    // Step 5: Apply inverse FFT to the result
    let ifft_result = ifftn(&mut multiplied_result);

    // Step 6: Center the result and remove near-zero values
    let center_offset = (result_len_usize - 1) / 2;
    let mut combined_stencil = HashMap::new();

    for (i, value) in ifft_result.iter().enumerate() {
        let offset = i as i32 - center_offset as i32;
        if value.abs() > EPSILON { // Remove near-zero values
            combined_stencil.insert(offset, *value);
        }
    }

    combined_stencil
}

// Define a type for the time-varying weight function
type TimeVaryingWeightFn = Box<dyn Fn(f64) -> f64>;

// Struct to represent Michael's hashmap based stencil which supports time varying weights
struct Stencil {
    // Mapping of d-dimensional coordinates (as Vec<i32>) to (constant weight, time-varying weight function)
    pub stencil_map: HashMap<Vec<i32>, (f64, Option<TimeVaryingWeightFn>)>,
}

// Implementation of stencil object supports time varying weights
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
    fn get_weight_at(&self, coords: &Vec<i32>, time: f64) -> f64 {
        if let Some((constant, Some(time_varying_fn))) = self.stencil_map.get(coords) {
            constant + time_varying_fn(time) // Apply time-varying function
        } else if let Some((constant, None)) = self.stencil_map.get(coords) {
            *constant // Return constant if no time-varying weight
        } else {
            0.0 // Return 0 if the coordinate is not found (default case)
        }
    }

    // Method to get the entire stencil at a specific time
    pub fn get_stencil_at(&self, time: f64) -> HashMap<Vec<i32>, f64> {
        let mut stencil_at_time = HashMap::new();
        
        // Iterate through all coordinates in the stencil map
        for (coords, _) in &self.stencil_map {
            // Get the weight at each coordinate at the specified time
            let weight = self.get_weight_at(coords, time);
            stencil_at_time.insert(coords.clone(), weight);
        }
        
        stencil_at_time
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


fn main() {

    //Creating the stencil
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

    // Some manual tests to make sure the get_weight_at and get_stencil_at function works
    println!("Weight at [-1] with t=2: {}", stencil.get_weight_at(&vec![-1], 2.0));
    println!("Weight at [0] with t=2: {}", stencil.get_weight_at(&vec![0], 2.0));
    println!("Weight at [1] with t=2: {}", stencil.get_weight_at(&vec![1], 2.0));
    println!("Weight at [2] with t=2: {}", stencil.get_weight_at(&vec![2], 2.0));

    println!("\n");

    let s1 = stencil.get_stencil_at(1.0);
    let s2 = stencil.get_stencil_at(2.0);
    let s3 = stencil.get_stencil_at(3.0);
    let s10 = stencil.get_stencil_at(10.0);
    println!("Stencil with t=1: {:?}", s1);
    println!("Stencil with t=2: {:?}", s2);
    println!("Stencil with t=3: {:?}", s3);
    println!("Stencil with t=10: {:?}", s10);

    // Some tests to make sure polymult functions work
    println!("Multiplied stencil at t=1 with stencil at t=1: {:?}", naive_mult(&s1, &s1))
}
