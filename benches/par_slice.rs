#![feature(test)]
extern crate test;
// Bench documentation
// https://doc.rust-lang.org/unstable-book/library-features/test.html

use fftw::types::*;
#[cfg(test)]
use test::Bencher;

use nhls::par_slice;

#[bench]
fn bench_power_01(bencher: &mut Bencher) {
    let chunk_size = 50000;
    let n = 20000000;
    let v = c32::new(1.1, 0.88);
    let mut a = vec![v; n];
    let mut b = vec![v; n];
    bencher.iter(|| {
        par_slice::power(132, &mut a, &mut b, chunk_size);
        par_slice::set_value(&mut a, v, chunk_size);
    });
    println!("vo: {:?}", b[0]);
}

#[bench]
fn bench_power_02(bencher: &mut Bencher) {
    let n = 200000000;
    let chunk_size = n;
    let v = c32::new(1.1, 0.88);
    let mut a = vec![v; n];
    let mut b = vec![v; n];
    bencher.iter(|| {
        par_slice::power(132, &mut a, &mut b, chunk_size);
        par_slice::set_value(&mut a, v, chunk_size);
    });
    println!("vo: {:?}", b[0]);
}
