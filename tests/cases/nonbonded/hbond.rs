use crate::verify_hybrid_potential;
use dreid_kernel::potentials::nonbonded::HydrogenBond;

verify_hybrid_potential!(
    HydrogenBond<4>,
    (1.0, 4.0),
    test_hbond_n4,
    cases: [
        (2.0, 0.0, -1.0),
        (2.0, 90.0, 0.0),
        (2.0, 45.0, -0.25)
    ]
);
