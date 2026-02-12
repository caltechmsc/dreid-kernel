use crate::bench_angle_potential;
use criterion::Criterion;
use dreid_kernel::potentials::bonded::{PlanarInversion, UmbrellaInversion};

pub fn bench_inversion(c: &mut Criterion) {
    bench_angle_potential!(
        c,
        "bonded/inversion/planar",
        PlanarInversion,
        PlanarInversion::precompute(40.0),
        80.0
    );

    bench_angle_potential!(
        c,
        "bonded/inversion/umbrella",
        UmbrellaInversion,
        UmbrellaInversion::precompute(40.0, 0.0),
        10.0
    );
}
