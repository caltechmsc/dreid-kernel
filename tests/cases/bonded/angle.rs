use crate::verify_angle_potential;
use dreid_kernel::potentials::bonded::{CosineHarmonic, ThetaHarmonic};

verify_angle_potential!(
    CosineHarmonic,
    (50.0, 0.0),
    test_cosine_harmonic,
    cases: [
        (90.0, 0.0),
        (0.0, 50.0),
        (180.0, 50.0),
        (60.0, 12.5)
    ]
);

use std::f64::consts::PI;
verify_angle_potential!(
    ThetaHarmonic,
    (50.0, PI / 2.0),
    test_theta_harmonic,
    cases: [
        (90.0, 0.0),
        (147.295779513, 50.0)
    ]
);
