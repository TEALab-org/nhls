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

///n^1.5 stencil mulitplication
pub fn karatsuba(s1: HashMap<Vec<i32>, f64>, s2: HashMap<Vec<i32>, f64>){

}

///Quadratic stencil mulitplication
pub fn fft_mult(s1: HashMap<Vec<i32>, f64>, s2: HashMap<Vec<i32>, f64>){

}