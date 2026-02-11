use crate::bench_pair_potential;
use criterion::Criterion;
use dreid_kernel::potentials::nonbonded::{Buckingham, LennardJones};

pub fn bench_vdw(c: &mut Criterion) {
    bench_pair_potential!(c, "non-bonded/lennard-jones", LennardJones, (1.0, 4.0), 4.0);

    bench_pair_potential!(
        c,
        "non-bonded/buckingham",
        Buckingham,
        (
            162754.79141900392,
            6.0,
            128.0,
            0.3659637525866052,
            3410.825177656595
        ),
        4.0
    );
}
