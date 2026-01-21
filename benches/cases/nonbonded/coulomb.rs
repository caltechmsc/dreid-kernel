use crate::bench_pair_potential;
use criterion::Criterion;
use dreid_kernel::potentials::nonbonded::Coulomb;

pub fn bench_coulomb(c: &mut Criterion) {
    bench_pair_potential!(c, "non-bonded/coulomb", Coulomb, -83.0, 3.5);
}
