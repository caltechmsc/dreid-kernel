use crate::bench_angle_potential;
use criterion::Criterion;
use dreid_kernel::potentials::bonded::{CosineHarmonic, CosineLinear, ThetaHarmonic};

pub fn bench_angle(c: &mut Criterion) {
    bench_angle_potential!(
        c,
        "bonded/angle/cosine-harmonic",
        CosineHarmonic,
        CosineHarmonic::precompute(100.0, 120.0),
        125.0
    );

    bench_angle_potential!(c, "bonded/angle/cosine-linear", CosineLinear, 100.0, 125.0);

    bench_angle_potential!(
        c,
        "bonded/angle/theta-harmonic",
        ThetaHarmonic,
        ThetaHarmonic::precompute(100.0, 120.0),
        125.0
    );
}
