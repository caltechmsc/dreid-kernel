use crate::bench_angle_potential;
use criterion::Criterion;
use dreid_kernel::potentials::bonded::{CosineHarmonic, ThetaHarmonic};

pub fn bench_angle(c: &mut Criterion) {
    bench_angle_potential!(
        c,
        "bonded/angle/cosine-harmonic",
        CosineHarmonic,
        (50.0, -0.5),
        125.0
    );

    bench_angle_potential!(
        c,
        "bonded/angle/theta-harmonic",
        ThetaHarmonic,
        (50.0, 2.09439510239),
        125.0
    );
}
