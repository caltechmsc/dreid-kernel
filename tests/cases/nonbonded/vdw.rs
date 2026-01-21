use crate::verify_pair_potential;
use dreid_kernel::potentials::nonbonded::{Buckingham, LennardJones};

verify_pair_potential!(
    LennardJones,
    (1.0, 4.0),
    test_lennard_jones,
    cases: [
        (2.0, -1.0),
        (4.0, -0.031005859375)
    ]
);

verify_pair_potential!(
    Buckingham,
    (162754.791419, 6.0, 128.0, 0.25),
    test_buckingham,
    cases: [
        (2.0, -1.0)
    ]
);

// Note: SplinedBuckingham requires complex params (Newton-Raphson method and matrix for solving the spline coefficients), so we skip testing it here.
