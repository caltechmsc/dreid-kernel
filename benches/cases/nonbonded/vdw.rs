use crate::bench_pair_potential;
use criterion::Criterion;
use dreid_kernel::potentials::nonbonded::{Buckingham, LennardJones, SplinedBuckingham};

pub fn bench_vdw(c: &mut Criterion) {
    bench_pair_potential!(c, "non-bonded/lennard-jones", LennardJones, (1.0, 4.0), 4.0);

    bench_pair_potential!(
        c,
        "non-bonded/buckingham",
        Buckingham,
        (162754.79141900392, 6.0, 128.0, 0.25),
        4.0
    );

    let dummy_params = (1000.0, 5.0, 100.0, 1.0, -100.0, 0.0, 10.0, 0.0, 0.0, 0.0);

    bench_pair_potential!(
        c,
        "non-bonded/splined-buckingham/long-range",
        SplinedBuckingham,
        dummy_params,
        2.0
    );

    bench_pair_potential!(
        c,
        "non-bonded/splined-buckingham/short-range",
        SplinedBuckingham,
        dummy_params,
        0.5
    );
}
