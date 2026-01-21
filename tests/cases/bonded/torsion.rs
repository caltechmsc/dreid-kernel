use crate::verify_torsion_potential;
use dreid_kernel::potentials::bonded::Torsion;

verify_torsion_potential!(
    Torsion,
    (2.0, 3, 1.0, 0.0),
    test_torsion_n3,
    cases: [
        (0.0, 0.0),
        (60.0, 4.0),
        (120.0, 0.0)
    ]
);
