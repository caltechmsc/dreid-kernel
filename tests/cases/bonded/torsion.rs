use crate::verify_torsion_potential;
use dreid_kernel::potentials::bonded::Torsion;

verify_torsion_potential!(
    Torsion,
    Torsion::precompute(4.0, 3, 0.0),
    test_torsion_n3,
    cases: [
        (0.0, 0.0),
        (60.0, 4.0),
        (120.0, 0.0)
    ]
);
