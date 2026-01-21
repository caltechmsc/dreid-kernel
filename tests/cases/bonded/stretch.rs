use crate::verify_pair_potential;
use dreid_kernel::potentials::bonded::{Harmonic, Morse};

verify_pair_potential!(
    Harmonic,
    (350.0, 1.53),
    test_harmonic_stretch,
    cases: [
        (1.53, 0.0),
        (1.63, 3.5),
        (1.43, 3.5)
    ]
);

verify_pair_potential!(
    Morse,
    (100.0, 1.53, 2.0),
    test_morse_stretch,
    cases: [
        (1.53, 0.0),
        (100.0, 100.0)
    ]
);
