use crate::math::Real;
use crate::traits::HybridKernel;
use crate::types::HybridEnergyDiff;

/// DREIDING Hydrogen Bond potential (12-10).
///
/// # Physics
///
/// Models the explicit hydrogen bonding interaction (typically D-H...A) using a
/// specific 12-10 Radial potential modulated by a $\cos^N\theta$ Angular term.
/// Standard DREIDING uses $N=4$.
///
/// - **Formula**: $$ E = D_{hb} \left[ 5 \left(\frac{R_{hb}}{r}\right)^{12} - 6 \left(\frac{R_{hb}}{r}\right)^{10} \right] \cos^N \theta $$
/// - **Derivative Factor (Radial)**: $$ D_{rad} = - \frac{1}{r} \frac{\partial E}{\partial r} = \frac{60 D_{hb}}{r^2} \left[ \left(\frac{R_{hb}}{r}\right)^{12} - \left(\frac{R_{hb}}{r}\right)^{10} \right] \cos^N \theta $$
/// - **Derivative Factor (Angular)**: $$ D_{ang} = \frac{\partial E}{\partial (\cos\theta)} = N \cdot E_{rad} \cos^{N-1} \theta $$
///
/// # Parameters
///
/// - `d_hb`: The energy well depth $D_{hb}$.
/// - `r_hb_sq`: The squared equilibrium distance $R_{hb}^2$.
/// - `N`: The cosine power exponent (const generic).
///
/// # Inputs
///
/// - `r_sq`: Squared distance $r^2$ between Donor (D) and Acceptor (A).
/// - `cos_theta`: Cosine of the angle $\theta_{DHA}$ (at Hydrogen).
///
/// # Implementation Notes
///
/// - **Cutoff**: If $\cos\theta \le 0$, energy and forces represent 0.
/// - **Optimization**: Uses $s = (R_{hb}/r)^2$ recurrence to compute $r^{-10}$ and $r^{-12}$ efficiently.
/// - **Generics**: Uses `const N: usize` to unroll power calculations at compile time.
#[derive(Clone, Copy, Debug, Default)]
pub struct HydrogenBond<const N: usize>;

impl<T: Real, const N: usize> HybridKernel<T> for HydrogenBond<N> {
    type Params = (T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = D_{hb} (5s^6 - 6s^5) \cos^N \theta, \quad \text{where } s = (R_{hb}/r)^2 $$
    #[inline(always)]
    fn energy(r_sq: T, cos_theta: T, (d_hb, r_hb_sq): Self::Params) -> T {
        let effective_cos = cos_theta.max(T::from(0.0));

        let cos_n = pow_n_helper(effective_cos, N);

        let s = r_hb_sq * r_sq.recip();
        let s2 = s * s;
        let s4 = s2 * s2;
        let s5 = s4 * s;
        let s6 = s4 * s2;

        let term12 = T::from(5.0) * s6;
        let term10 = T::from(6.0) * s5;

        (d_hb * (term12 - term10)) * cos_n
    }

    /// Computes only the derivative factors.
    ///
    /// # Formula
    ///
    /// $$ D_{rad} = \frac{60 D_{hb}}{r^2} (s^6 - s^5) \cos^N \theta, \quad \text{where } s = (R_{hb}/r)^2 $$
    /// $$ D_{ang} = N E_{rad} \cos^{N-1} \theta $$
    ///
    /// - `force_factor_rad` ($D_{rad}$): Used to compute the central force along the D-A axis:
    ///   $ \vec{F}_{rad} = -D\_{rad} \cdot \vec{r}\_{DA} $
    /// - `force_factor_ang` ($D_{ang}$): Used to compute torque-like forces on the D-H-A angle
    ///   via the Wilson B-matrix gradient chain rule:
    ///   $ \vec{F}_i = -D\_{ang} \cdot \nabla_i (\cos\theta) $
    #[inline(always)]
    fn diff(r_sq: T, cos_theta: T, (d_hb, r_hb_sq): Self::Params) -> (T, T) {
        let effective_cos = cos_theta.max(T::from(0.0));

        let inv_r2 = r_sq.recip();
        let s = r_hb_sq * inv_r2;
        let s2 = s * s;
        let s4 = s2 * s2;
        let s5 = s4 * s;
        let s6 = s4 * s2;

        let term12 = T::from(5.0) * s6;
        let term10 = T::from(6.0) * s5;
        let e_rad_pure = d_hb * (term12 - term10);

        let cos_n_minus_1 = if N == 0 {
            T::from(0.0)
        } else if N == 1 {
            T::from(1.0)
        } else {
            pow_n_helper(effective_cos, N - 1)
        };

        let cos_n = if N == 0 {
            T::from(1.0)
        } else {
            cos_n_minus_1 * effective_cos
        };

        let diff_rad_pure = T::from(60.0) * d_hb * inv_r2 * (s6 - s5);
        let force_factor_rad = diff_rad_pure * cos_n;

        let force_factor_ang = T::from(N as f32) * e_rad_pure * cos_n_minus_1;

        (force_factor_rad, force_factor_ang)
    }

    /// Computes both energy and derivative factors efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(r_sq: T, cos_theta: T, (d_hb, r_hb_sq): Self::Params) -> HybridEnergyDiff<T> {
        let effective_cos = cos_theta.max(T::from(0.0));

        let inv_r2 = r_sq.recip();
        let s = r_hb_sq * inv_r2;
        let s2 = s * s;
        let s4 = s2 * s2;
        let s5 = s4 * s;
        let s6 = s4 * s2;

        let term12 = T::from(5.0) * s6;
        let term10 = T::from(6.0) * s5;
        let e_rad_pure = d_hb * (term12 - term10);

        let cos_n_minus_1 = if N == 0 {
            T::from(0.0)
        } else if N == 1 {
            T::from(1.0)
        } else {
            pow_n_helper(effective_cos, N - 1)
        };

        let cos_n = if N == 0 {
            T::from(1.0)
        } else {
            cos_n_minus_1 * effective_cos
        };

        let energy = e_rad_pure * cos_n;

        let diff_rad_pure = T::from(60.0) * d_hb * inv_r2 * (s6 - s5);
        let force_factor_rad = diff_rad_pure * cos_n;

        let force_factor_ang = T::from(N as f32) * e_rad_pure * cos_n_minus_1;

        HybridEnergyDiff {
            energy,
            force_factor_rad,
            force_factor_ang,
        }
    }
}

/// Helper to compute x^n using explicit unrolling for small common powers,
/// and fast exponentiation for larger n.
#[inline(always)]
fn pow_n_helper<T: Real>(base: T, n: usize) -> T {
    match n {
        0 => T::from(1.0),
        1 => base,
        2 => base * base,
        3 => base * base * base,
        4 => {
            let x2 = base * base;
            x2 * x2
        }
        5 => {
            let x2 = base * base;
            let x4 = x2 * x2;
            x4 * base
        }
        6 => {
            let x2 = base * base;
            let x4 = x2 * x2;
            x4 * x2
        }
        _ => {
            let mut acc = T::from(1.0);
            let mut b = base;
            let mut e = n;
            while e > 0 {
                if e & 1 == 1 {
                    acc = acc * b;
                }
                b = b * b;
                e >>= 1;
            }
            acc
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ------------------------------------------------------------------------
    // Test Constants
    // ------------------------------------------------------------------------

    const H: f64 = 1e-6;
    const TOL_DIFF: f64 = 1e-4;

    // Typical H-bond parameters
    const D_HB: f64 = 4.0; // kcal/mol
    const R_HB: f64 = 2.75; // Å
    const R_HB_SQ: f64 = R_HB * R_HB;

    fn params() -> (f64, f64) {
        (D_HB, R_HB_SQ)
    }

    // ========================================================================
    // HydrogenBond<4> Tests (Standard DREIDING)
    // ========================================================================

    mod hydrogen_bond_n4 {
        use super::*;

        type HBond4 = HydrogenBond<4>;

        // --------------------------------------------------------------------
        // 1. Sanity Checks
        // --------------------------------------------------------------------

        #[test]
        fn sanity_compute_equals_separate() {
            let r_sq = 9.0_f64;
            let cos_theta = 0.9;
            let p = params();

            let result = HBond4::compute(r_sq, cos_theta, p);
            let energy_only = HBond4::energy(r_sq, cos_theta, p);
            let (rad_only, ang_only) = HBond4::diff(r_sq, cos_theta, p);

            assert_relative_eq!(result.energy, energy_only, epsilon = 1e-12);
            assert_relative_eq!(result.force_factor_rad, rad_only, epsilon = 1e-12);
            assert_relative_eq!(result.force_factor_ang, ang_only, epsilon = 1e-12);
        }

        #[test]
        fn sanity_f32_f64_consistency() {
            let r_sq = 9.0;
            let cos_theta = 0.8;
            let p64 = params();
            let p32 = (D_HB as f32, R_HB_SQ as f32);

            let e64 = HBond4::energy(r_sq, cos_theta, p64);
            let e32 = HBond4::energy(r_sq as f32, cos_theta as f32, p32);

            assert_relative_eq!(e64, e32 as f64, epsilon = 1e-4);
        }

        #[test]
        fn sanity_equilibrium_energy_minimum() {
            let e = HBond4::energy(R_HB_SQ, 1.0, params());
            assert_relative_eq!(e, -D_HB, epsilon = 1e-10);
        }

        // --------------------------------------------------------------------
        // 2. Numerical Stability
        // --------------------------------------------------------------------

        #[test]
        fn stability_cos_theta_zero() {
            let result = HBond4::compute(R_HB_SQ, 0.0, params());

            assert!(result.energy.is_finite());
            assert_relative_eq!(result.energy, 0.0, epsilon = 1e-14);
            assert_relative_eq!(result.force_factor_rad, 0.0, epsilon = 1e-14);
        }

        #[test]
        fn stability_cos_theta_negative() {
            let result = HBond4::compute(R_HB_SQ, -0.5, params());

            assert!(result.energy.is_finite());
            assert_relative_eq!(result.energy, 0.0, epsilon = 1e-14);
        }

        #[test]
        fn stability_large_distance() {
            let result = HBond4::compute(1e4, 1.0, params());

            assert!(result.energy.is_finite());
            assert!(result.force_factor_rad.is_finite());
            assert!(result.energy.abs() < 1e-10);
        }

        // --------------------------------------------------------------------
        // 3. Finite Difference Verification
        // --------------------------------------------------------------------

        fn finite_diff_radial(r: f64, cos_theta: f64) {
            let p = params();
            let r_sq = r * r;

            let r_plus = r + H;
            let r_minus = r - H;
            let e_plus = HBond4::energy(r_plus * r_plus, cos_theta, p);
            let e_minus = HBond4::energy(r_minus * r_minus, cos_theta, p);
            let de_dr_numerical = (e_plus - e_minus) / (2.0 * H);

            let (d_rad, _) = HBond4::diff(r_sq, cos_theta, p);
            let de_dr_analytic = -d_rad * r;

            assert_relative_eq!(de_dr_numerical, de_dr_analytic, epsilon = TOL_DIFF);
        }

        fn finite_diff_angular(r_sq: f64, cos_theta: f64) {
            let p = params();

            let c_plus = cos_theta + H;
            let c_minus = cos_theta - H;
            let e_plus = HBond4::energy(r_sq, c_plus, p);
            let e_minus = HBond4::energy(r_sq, c_minus, p);
            let de_dcos_numerical = (e_plus - e_minus) / (2.0 * H);

            let (_, d_ang) = HBond4::diff(r_sq, cos_theta, p);

            assert_relative_eq!(de_dcos_numerical, d_ang, epsilon = TOL_DIFF);
        }

        #[test]
        fn finite_diff_radial_short() {
            finite_diff_radial(2.0, 0.9);
        }

        #[test]
        fn finite_diff_radial_equilibrium() {
            finite_diff_radial(R_HB, 0.8);
        }

        #[test]
        fn finite_diff_radial_long() {
            finite_diff_radial(5.0, 0.95);
        }

        #[test]
        fn finite_diff_angular_strong() {
            finite_diff_angular(R_HB_SQ, 0.9);
        }

        #[test]
        fn finite_diff_angular_weak() {
            finite_diff_angular(R_HB_SQ, 0.3);
        }

        // --------------------------------------------------------------------
        // 4. H-Bond Specific
        // --------------------------------------------------------------------

        #[test]
        fn specific_cos4_scaling() {
            let e1 = HBond4::energy(R_HB_SQ, 1.0, params());
            let e_half = HBond4::energy(R_HB_SQ, 0.5, params());

            assert_relative_eq!(e1 / e_half, 16.0, epsilon = 1e-10);
        }

        #[test]
        fn specific_12_10_radial_form() {
            let e_large = HBond4::energy(100.0, 1.0, params());
            let e_small = HBond4::energy(1.0, 1.0, params());

            assert!(e_large < 0.0);
            assert!(e_small > 0.0);
        }
    }

    // ========================================================================
    // HydrogenBond<0> Tests (No Angular Dependence)
    // ========================================================================

    mod hydrogen_bond_n0 {
        use super::*;

        type HBond0 = HydrogenBond<0>;

        #[test]
        fn n0_energy_independent_of_angle() {
            let p = params();
            let e1 = HBond0::energy(R_HB_SQ, 0.0, p);
            let e2 = HBond0::energy(R_HB_SQ, 0.5, p);
            let e3 = HBond0::energy(R_HB_SQ, 1.0, p);

            assert_relative_eq!(e1, e2, epsilon = 1e-14);
            assert_relative_eq!(e2, e3, epsilon = 1e-14);
        }

        #[test]
        fn n0_angular_derivative_zero() {
            let (_, d_ang) = HBond0::diff(R_HB_SQ, 0.8, params());
            assert_relative_eq!(d_ang, 0.0, epsilon = 1e-14);
        }
    }

    mod pow_helper {
        use super::*;

        #[test]
        fn pow_n_0() {
            assert_relative_eq!(pow_n_helper(5.0_f64, 0), 1.0, epsilon = 1e-14);
        }

        #[test]
        fn pow_n_1() {
            assert_relative_eq!(pow_n_helper(5.0_f64, 1), 5.0, epsilon = 1e-14);
        }

        #[test]
        fn pow_n_4() {
            assert_relative_eq!(pow_n_helper(2.0_f64, 4), 16.0, epsilon = 1e-14);
        }

        #[test]
        fn pow_n_large() {
            assert_relative_eq!(pow_n_helper(2.0_f64, 10), 1024.0, epsilon = 1e-10);
        }
    }
}
