#[macro_export]
macro_rules! bench_pair_potential {
    (
        $c:expr,
        $group_name:expr,
        $kernel_type:ty,
        $params:expr,
        $r_val:expr
    ) => {
        let mut group = $c.benchmark_group($group_name);
        let params = $params;
        let r = $r_val;
        let r_sq = r * r;

        group.bench_function("energy", |b| {
            b.iter(|| {
                use std::hint::black_box;
                <$kernel_type as dreid_kernel::PairKernel<f64>>::energy(
                    black_box(r_sq),
                    black_box(params),
                )
            })
        });

        group.bench_function("diff", |b| {
            b.iter(|| {
                use std::hint::black_box;
                <$kernel_type as dreid_kernel::PairKernel<f64>>::diff(
                    black_box(r_sq),
                    black_box(params),
                )
            })
        });

        group.bench_function("compute", |b| {
            b.iter(|| {
                use std::hint::black_box;
                <$kernel_type as dreid_kernel::PairKernel<f64>>::compute(
                    black_box(r_sq),
                    black_box(params),
                )
            })
        });

        group.finish();
    };
}

#[macro_export]
macro_rules! bench_angle_potential {
    (
        $c:expr,
        $group_name:expr,
        $kernel_type:ty,
        $params:expr,
        $deg_val:expr
    ) => {
        let mut group = $c.benchmark_group($group_name);
        let params = $params;
        let deg: f64 = $deg_val;
        let cos_theta = deg.to_radians().cos();

        group.bench_function("energy", |b| {
            b.iter(|| {
                use std::hint::black_box;
                <$kernel_type as dreid_kernel::AngleKernel<f64>>::energy(
                    black_box(cos_theta),
                    black_box(params),
                )
            })
        });

        group.bench_function("diff", |b| {
            b.iter(|| {
                use std::hint::black_box;
                <$kernel_type as dreid_kernel::AngleKernel<f64>>::diff(
                    black_box(cos_theta),
                    black_box(params),
                )
            })
        });

        group.bench_function("compute", |b| {
            b.iter(|| {
                use std::hint::black_box;
                <$kernel_type as dreid_kernel::AngleKernel<f64>>::compute(
                    black_box(cos_theta),
                    black_box(params),
                )
            })
        });

        group.finish();
    };
}

#[macro_export]
macro_rules! bench_torsion_potential {
    (
        $c:expr,
        $group_name:expr,
        $kernel_type:ty,
        $params:expr,
        $deg_val:expr
    ) => {
        let mut group = $c.benchmark_group($group_name);
        let params = $params;
        let deg: f64 = $deg_val;
        let (sin_phi, cos_phi) = deg.to_radians().sin_cos();

        group.bench_function("energy", |b| {
            b.iter(|| {
                use std::hint::black_box;
                <$kernel_type as dreid_kernel::TorsionKernel<f64>>::energy(
                    black_box(cos_phi),
                    black_box(sin_phi),
                    black_box(params),
                )
            })
        });

        group.bench_function("diff", |b| {
            b.iter(|| {
                use std::hint::black_box;
                <$kernel_type as dreid_kernel::TorsionKernel<f64>>::diff(
                    black_box(cos_phi),
                    black_box(sin_phi),
                    black_box(params),
                )
            })
        });

        group.bench_function("compute", |b| {
            b.iter(|| {
                use std::hint::black_box;
                <$kernel_type as dreid_kernel::TorsionKernel<f64>>::compute(
                    black_box(cos_phi),
                    black_box(sin_phi),
                    black_box(params),
                )
            })
        });

        group.finish();
    };
}

#[macro_export]
macro_rules! bench_hybrid_potential {
    (
        $c:expr,
        $group_name:expr,
        $kernel_type:ty,
        $params:expr,
        $r_val:expr,
        $deg_val:expr
    ) => {
        let mut group = $c.benchmark_group($group_name);
        let params = $params;
        let r_sq = $r_val * $r_val;
        let cos_theta = ($deg_val as f64).to_radians().cos();

        group.bench_function("energy", |b| {
            b.iter(|| {
                use std::hint::black_box;
                <$kernel_type as dreid_kernel::HybridKernel<f64>>::energy(
                    black_box(r_sq),
                    black_box(cos_theta),
                    black_box(params),
                )
            })
        });

        group.bench_function("diff", |b| {
            b.iter(|| {
                use std::hint::black_box;
                <$kernel_type as dreid_kernel::HybridKernel<f64>>::diff(
                    black_box(r_sq),
                    black_box(cos_theta),
                    black_box(params),
                )
            })
        });

        group.bench_function("compute", |b| {
            b.iter(|| {
                use std::hint::black_box;
                <$kernel_type as dreid_kernel::HybridKernel<f64>>::compute(
                    black_box(r_sq),
                    black_box(cos_theta),
                    black_box(params),
                )
            })
        });

        group.finish();
    };
}
