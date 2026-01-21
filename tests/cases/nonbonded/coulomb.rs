use crate::verify_pair_potential;
use dreid_kernel::potentials::nonbonded::Coulomb;

verify_pair_potential!(
    Coulomb,
    332.0,
    test_coulomb,
    cases: [
        (1.0, 332.0),
        (2.0, 166.0)
    ]
);
