use criterion::{criterion_group, criterion_main};

mod cases;
mod common;

use cases::bonded::angle::bench_angle;
use cases::bonded::inversion::bench_inversion;
use cases::bonded::stretch::bench_stretch;
use cases::bonded::torsion::bench_torsion;
use cases::nonbonded::coulomb::bench_coulomb;
use cases::nonbonded::hbond::bench_hbond;
use cases::nonbonded::vdw::bench_vdw;

criterion_group!(
    benches,
    bench_vdw,
    bench_coulomb,
    bench_hbond,
    bench_stretch,
    bench_angle,
    bench_inversion,
    bench_torsion
);
criterion_main!(benches);
