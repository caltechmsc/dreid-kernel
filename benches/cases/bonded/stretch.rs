use crate::bench_pair_potential;
use criterion::Criterion;
use dreid_kernel::potentials::bonded::{Harmonic, Morse};

pub fn bench_stretch(c: &mut Criterion) {
    bench_pair_potential!(c, "bonded/stretch/harmonic", Harmonic, (150.0, 1.5), 1.6);

    bench_pair_potential!(c, "bonded/stretch/morse", Morse, (100.0, 2.0, 1.5), 1.6);
}
