use crate::verify_angle_potential;
use dreid_kernel::potentials::bonded::{PlanarInversion, UmbrellaInversion};

verify_angle_potential!(
    PlanarInversion,
    PlanarInversion::precompute(40.0),
    test_planar_inversion,
    cases: [
        (90.0, 0.0),
        (0.0, 20.0)
    ]
);

verify_angle_potential!(
    UmbrellaInversion,
    UmbrellaInversion::precompute(100.0, 0.0),
    test_umbrella_inversion,
    cases: [
        (0.0, 0.0),
        (60.0, 12.5),
        (90.0, 50.0)
    ]
);
