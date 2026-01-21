use crate::bench_torsion_potential;
use criterion::Criterion;
use dreid_kernel::potentials::bonded::Torsion;

pub fn bench_torsion(c: &mut Criterion) {
    bench_torsion_potential!(c, "bonded/torsion/n1", Torsion, (5.0, 1, 1.0, 0.0), 60.0);

    bench_torsion_potential!(c, "bonded/torsion/n3", Torsion, (5.0, 3, 1.0, 0.0), 60.0);

    bench_torsion_potential!(c, "bonded/torsion/n6", Torsion, (5.0, 6, 1.0, 0.0), 60.0);
}
