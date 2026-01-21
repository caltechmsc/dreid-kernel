use crate::bench_hybrid_potential;
use criterion::Criterion;
use dreid_kernel::potentials::nonbonded::HydrogenBond;

pub fn bench_hbond(c: &mut Criterion) {
    bench_hybrid_potential!(
        c,
        "non-bonded/h-bond/n4",
        HydrogenBond<4>,
        (5.0, 4.0),
        2.5,
        20.0
    );
}
