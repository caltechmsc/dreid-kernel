use crate::verify_angle_potential;
use dreid_kernel::potentials::bonded::{CosineHarmonic, CosineLinear, ThetaHarmonic};

verify_angle_potential!(
    CosineHarmonic,
    CosineHarmonic::precompute(100.0, 90.0),
    test_cosine_harmonic,
    cases: [
        (90.0, 0.0),
        (0.0, 50.0),
        (180.0, 50.0),
        (60.0, 12.5)
    ]
);

verify_angle_potential!(
    CosineLinear,
    100.0,
    test_cosine_linear,
    cases: [
        (180.0, 0.0),
        (0.0, 200.0),
        (90.0, 100.0),
        (120.0, 50.0)
    ]
);

verify_angle_potential!(
    ThetaHarmonic,
    ThetaHarmonic::precompute(100.0, 90.0),
    test_theta_harmonic,
    cases: [
        (90.0, 0.0),
        (147.295779513, 50.0)
    ]
);
