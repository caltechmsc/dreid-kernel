use crate::verify_pair_potential;
use dreid_kernel::potentials::nonbonded::{Buckingham, LennardJones};

verify_pair_potential!(
    LennardJones,
    LennardJones::precompute(1.0, 2.0),
    test_lennard_jones,
    cases: [
        (2.0, -1.0),
        (4.0, -0.031005859375)
    ]
);

verify_pair_potential!(
    Buckingham,
    Buckingham::precompute(1.0, 2.0, 12.0),
    test_buckingham,
    cases: [
        (2.0, -1.0),
        (0.4, 19896.043612)
    ]
);
