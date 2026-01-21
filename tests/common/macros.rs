#[macro_export]
macro_rules! verify_pair_potential {
    (
        $kernel_type:ty,
        $params:expr,
        $mod_name:ident,
        cases: [ $( ($r_val:expr, $e_expected:expr) ),* ]
    ) => {
        mod $mod_name {
            use super::*;
            use dreid_kernel::PairKernel;
            use approx::assert_relative_eq;

            #[test]
            fn test_stability() {
                let params = $params;
                let r_near_zero: f64 = 1e-10;
                let e_zero: f64 = <$kernel_type>::energy(r_near_zero * r_near_zero, params);
                assert!(!e_zero.is_nan(), "Energy at r=1e-10 should not be NaN");

                let mut r = 0.5f64;
                while r < 10.0 {
                    let r_sq = r * r;
                    let e = <$kernel_type>::energy(r_sq, params);
                    let d = <$kernel_type>::diff(r_sq, params);
                    let both = <$kernel_type>::compute(r_sq, params);

                    assert!(!e.is_nan(), "Energy NaN at r={}", r);
                    assert!(!d.is_nan(), "Diff NaN at r={}", r);
                    assert_relative_eq!(e, both.energy, epsilon = 1e-10);
                    assert_relative_eq!(d, both.diff, epsilon = 1e-10);

                    r += 0.1;
                }
            }

            #[test]
            fn test_numerical_derivative() {
                let params = $params;
                let test_points = [0.8, 1.0, 1.122, 1.5, 2.5, 5.0];

                for &r in &test_points {
                    let r_sq = r * r;

                    let d_analytical = <$kernel_type>::diff(r_sq, params);

                    let h = 1e-5;
                    let r_plus = r + h;
                    let r_minus = r - h;

                    let e_plus = <$kernel_type>::energy(r_plus * r_plus, params);
                    let e_minus = <$kernel_type>::energy(r_minus * r_minus, params);

                    let de_dr_num = (e_plus - e_minus) / (2.0 * h);
                    let d_numerical = - (1.0 / r) * de_dr_num;

                    assert_relative_eq!(
                        d_analytical,
                        d_numerical,
                        epsilon = 1e-4,
                        max_relative = 1e-4
                    );
                }
            }

            #[test]
            fn test_gold_standard_values() {
                let params = $params;
                $(
                    let r = $r_val;
                    let expected = $e_expected;
                    let r_sq = r * r;
                    let actual = <$kernel_type>::energy(r_sq, params);

                    assert_relative_eq!(
                        actual,
                        expected,
                        epsilon = 1e-5,
                        max_relative = 1e-5
                    );
                )*
            }
        }
    };
}

#[macro_export]
macro_rules! verify_angle_potential {
    (
        $kernel_type:ty,
        $params:expr,
        $mod_name:ident,
        cases: [ $( ($deg:expr, $e_expected:expr) ),* ]
    ) => {
        mod $mod_name {
            use super::*;
            use dreid_kernel::AngleKernel;
            use approx::assert_relative_eq;

            #[test]
            fn test_stability() {
                let params = $params;

                let steps = 20;
                for i in 0..=steps {
                    let c = -1.0 + (2.0 * i as f64 / steps as f64);
                    let cos_theta = c.clamp(-1.0, 1.0);

                    let e = <$kernel_type>::energy(cos_theta, params);
                    let d = <$kernel_type>::diff(cos_theta, params);

                    assert!(!e.is_nan(), "Energy NaN at cos={}", cos_theta);
                    assert!(!d.is_nan(), "Diff NaN at cos={}", cos_theta);
                }
            }

            #[test]
            fn test_numerical_derivative() {
                let params = $params;
                let test_angles_deg: [f64; 5] = [30.0, 60.0, 90.0, 120.0, 150.0];

                for &deg in &test_angles_deg {
                    let rad = deg.to_radians();
                    let cos_theta = rad.cos();

                    let d_analytical = <$kernel_type>::diff(cos_theta, params);

                    let h = 1e-5;

                    if (cos_theta + h) > 1.0 || (cos_theta - h) < -1.0 {
                        continue;
                    }

                    let e_plus = <$kernel_type>::energy(cos_theta + h, params);
                    let e_minus = <$kernel_type>::energy(cos_theta - h, params);

                    let d_numerical = (e_plus - e_minus) / (2.0 * h);

                    assert_relative_eq!(
                        d_analytical,
                        d_numerical,
                        epsilon = 1e-4,
                        max_relative = 1e-4
                    );
                }
            }

            #[test]
            fn test_gold_standard_values() {
                let params = $params;
                $(
                    let deg: f64 = $deg as f64;
                    let rad = deg.to_radians();
                    let cos_theta = rad.cos();
                    let expected = $e_expected;

                    let actual = <$kernel_type>::energy(cos_theta, params);
                    assert_relative_eq!(actual, expected, epsilon=1e-5);
                )*
            }
        }
    };
}

#[macro_export]
macro_rules! verify_torsion_potential {
    (
        $kernel_type:ty,
        $params:expr,
        $mod_name:ident,
        cases: [ $( ($deg:expr, $e_expected:expr) ),* ]
    ) => {
        mod $mod_name {
            use super::*;
            use dreid_kernel::TorsionKernel;
            use approx::assert_relative_eq;

            #[test]
            fn test_stability() {
                let params = $params;
                for i in -18..=18 {
                    let deg = i as f64 * 10.0;
                    let rad = deg.to_radians();
                    let c = rad.cos();
                    let s = rad.sin();

                    let e = <$kernel_type>::energy(c, s, params);
                    let d = <$kernel_type>::diff(c, s, params);

                    assert!(!e.is_nan(), "Energy NaN at phi={}", deg);
                    assert!(!d.is_nan(), "Torque NaN at phi={}", deg);
                }
            }

            #[test]
            fn test_numerical_derivative() {
                let params = $params;
                let test_angles: [f64; 6] = [-150.0, -90.0, -30.0, 30.0, 90.0, 150.0];

                for &deg in &test_angles {
                    let rad = deg.to_radians();
                    let c = rad.cos();
                    let s = rad.sin();

                    let d_analytical = <$kernel_type>::diff(c, s, params);

                    let h = 1e-5;

                    let rad_plus = rad + h;
                    let rad_minus = rad - h;

                    let e_plus = <$kernel_type>::energy(rad_plus.cos(), rad_plus.sin(), params);
                    let e_minus = <$kernel_type>::energy(rad_minus.cos(), rad_minus.sin(), params);

                    let d_numerical = (e_plus - e_minus) / (2.0 * h);

                    assert_relative_eq!(
                        d_analytical,
                        d_numerical,
                        epsilon = 1e-4,
                        max_relative = 1e-4
                    );
                }
            }

            #[test]
            fn test_gold_standard_values() {
                let params = $params;
                $(
                    let deg: f64 = $deg as f64;
                    let rad = deg.to_radians();
                    let c = rad.cos();
                    let s = rad.sin();
                    let expected = $e_expected;

                    let actual = <$kernel_type>::energy(c, s, params);
                    assert_relative_eq!(actual, expected, epsilon=1e-5);
                )*
            }
        }
    };
}

#[macro_export]
macro_rules! verify_hybrid_potential {
    (
        $kernel_type:ty,
        $params:expr,
        $mod_name:ident,
        cases: [ $( ($r_val:expr, $deg:expr, $e_expected:expr) ),* ]
    ) => {
        mod $mod_name {
            use super::*;
            use dreid_kernel::HybridKernel;
            use approx::assert_relative_eq;

            #[test]
            fn test_stability() {
                let params = $params;
                let r_vals: [f64; 3] = [1.5, 2.5, 4.0];
                let angles: [f64; 3] = [0.0, 90.0, 180.0];

                for &r in &r_vals {
                    for &deg in &angles {
                        let r_sq = r * r;
                        let cos_theta = deg.to_radians().cos();

                        let e = <$kernel_type>::energy(r_sq, cos_theta, params);
                        let (dr, da) = <$kernel_type>::diff(r_sq, cos_theta, params);

                        assert!(!e.is_nan(), "Energy NaN at r={}, ang={}", r, deg);
                        assert!(!dr.is_nan(), "Radial Diff NaN at r={}, ang={}", r, deg);
                        assert!(!da.is_nan(), "Angular Diff NaN at r={}, ang={}", r, deg);
                    }
                }
            }

            #[test]
            fn test_numerical_derivative() {
                let params = $params;
                let test_r = [2.0, 3.0, 4.0];
                let test_deg: [f64; 4] = [30.0, 90.0, 150.0, 120.0];

                for &r in &test_r {
                    for &deg in &test_deg {
                        let r_sq = r * r;
                        let rad = deg.to_radians();
                        let cos_theta = rad.cos();

                        let (dr_analytic, da_analytic) = <$kernel_type>::diff(r_sq, cos_theta, params);

                        let h = 1e-5;
                        let r_plus = r + h;
                        let r_minus = r - h;
                        let e_r_plus = <$kernel_type>::energy(r_plus * r_plus, cos_theta, params);
                        let e_r_minus = <$kernel_type>::energy(r_minus * r_minus, cos_theta, params);
                        let de_dr = (e_r_plus - e_r_minus) / (2.0 * h);
                        let dr_numerical = -(1.0/r) * de_dr;

                        if cos_theta.abs() > 0.9 { continue; }

                        let c_plus = cos_theta + h;
                        let c_minus = cos_theta - h;
                        let e_c_plus = <$kernel_type>::energy(r_sq, c_plus, params);
                        let e_c_minus = <$kernel_type>::energy(r_sq, c_minus, params);
                        let da_numerical = (e_c_plus - e_c_minus) / (2.0 * h);

                        assert_relative_eq!(dr_analytic, dr_numerical, epsilon = 1e-4);
                        assert_relative_eq!(da_analytic, da_numerical, epsilon = 1e-4);
                    }
                }
            }

            #[test]
            fn test_gold_standard_values() {
                let params = $params;
                $(
                    let r: f64 = $r_val;
                    let deg: f64 = $deg as f64;
                    let expected = $e_expected;

                    let r_sq = r * r;
                    let cos_theta = deg.to_radians().cos();
                    let actual = <$kernel_type>::energy(r_sq, cos_theta, params);

                    assert_relative_eq!(actual, expected, epsilon = 1e-5);
                )*
            }
        }
    };
}
