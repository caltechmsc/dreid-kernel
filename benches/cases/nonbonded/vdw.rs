use crate::bench_pair_potential;
use criterion::Criterion;
use dreid_kernel::potentials::nonbonded::{Buckingham, LennardJones};

pub fn bench_vdw(c: &mut Criterion) {
    bench_pair_potential!(
        c,
        "non-bonded/lennard-jones",
        LennardJones,
        LennardJones::precompute(1.0, 2.0),
        4.0
    );

    bench_pair_potential!(
        c,
        "non-bonded/buckingham",
        Buckingham,
        Buckingham::precompute(1.0, 2.0, 12.0),
        4.0
    );
}
