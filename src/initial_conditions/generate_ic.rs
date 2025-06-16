use crate::domain::*;
use crate::initial_conditions::normal_impulse::*;
use crate::initial_conditions::rand::*;
use crate::initial_conditions::*;

pub fn generate_ic_1d(
    domain: &mut SliceDomain<1>,
    ic_type: ICType,
    chunk_size: usize,
) {
    match ic_type {
        // Special case, domains are initializes to
        ICType::Zero => {}
        ICType::Rand { max_val } => {
            rand_ic(domain, max_val, chunk_size);
        }
        ICType::Impulse { variance } => {
            normal_ic_1d(domain, variance, chunk_size);
        }
    }
}

pub fn generate_ic_2d(
    domain: &mut SliceDomain<2>,
    ic_type: ICType,
    chunk_size: usize,
) {
    match ic_type {
        // Special case, domains are initializes to
        ICType::Zero => {}
        ICType::Rand { max_val } => {
            rand_ic(domain, max_val, chunk_size);
        }
        ICType::Impulse { variance } => {
            normal_ic_2d(domain, variance, chunk_size);
        }
    }
}

pub fn generate_ic_3d(
    domain: &mut SliceDomain<3>,
    ic_type: ICType,
    chunk_size: usize,
) {
    match ic_type {
        // Special case, domains are initializes to
        ICType::Zero => {}
        ICType::Rand { max_val } => {
            rand_ic(domain, max_val, chunk_size);
        }
        ICType::Impulse { variance } => {
            normal_ic_3d(domain, variance, chunk_size);
        }
    }
}
