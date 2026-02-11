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
    (162754.79141900392, 6.0, 128.0, 0.3659637525866052, 3410.825177656595),
    test_buckingham,
    cases: [
        (2.0, -1.0)
    ]
);
