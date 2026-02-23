use crate::arbreg::Region;
use crate::domain::*;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Domain {
    S1, // Remove energy 0.2
    S2, // Conserve Energy 0.3
    S3, // Add Energy 0.4
}

pub struct MovingBoundary {
    pub inside: Region,
    pub outside: Region,
}

pub struct ArbDomain {
    pub fixed: Option<Region>,
    pub moving: Option<MovingBoundary>,
}

pub struct DomainBoundaries {
   s1: Option<ArbDomain>,
   s2: Option<ArbDomain>,
   s3: Option<ArbDomain>,
}


pub fn mask_to_boundaries(mask: &OwnedDomainMask<2, Domain>) -> DomainBoundaries {
    // Iterate through each coord

    // Do the usual neighbor search

    // create all the inside boundaries

    DomainBoundaries {
        s1: None,
        s2: None,
        s3: None,
    }
}
