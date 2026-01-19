use crate::math::Real;
use crate::traits::TorsionKernel;
use crate::types::EnergyDiff;

/// Periodic torsion potential for dihedral angles.
///
/// # Physics
///
/// Models the rotational barrier around a bond axis using a periodic cosine function.
///
/// - **Formula**: $$ E = \frac{1}{2} V [1 - \cos(n(\phi - \phi_0))] $$
/// - **Derivative (`diff`)**: $$ T = \frac{dE}{d\phi} = \frac{1}{2} V \cdot n \cdot \sin(n(\phi - \phi_0)) $$
///
/// # Parameters
///
/// - `v_half`: Half barrier height $V_{half} = V/2$.
/// - `n`: Periodicity/multiplicity.
/// - `cos_n_phi0`: $\cos(n\phi_0)$, pre-computed phase cosine.
/// - `sin_n_phi0`: $\sin(n\phi_0)$, pre-computed phase sine.
///
/// # Inputs
///
/// - `cos_phi`: Cosine of the dihedral angle $\cos\phi$.
/// - `sin_phi`: Sine of the dihedral angle $\sin\phi$.
///
/// # Implementation Notes
///
/// - Uses optimized closed-form formulas for common periodicities ($n = 1, 2, 3$).
/// - Falls back to Chebyshev recurrence for higher periodicities.
/// - All intermediate calculations are shared between energy and torque computations.
/// - Branchless and panic-free.
#[derive(Clone, Copy, Debug, Default)]
pub struct Torsion;

impl<T: Real> TorsionKernel<T> for Torsion {
    type Params = (T, u8, T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = V_{half} [1 - \cos(n(\phi - \phi_0))] $$
    #[inline(always)]
    fn energy(cos_phi: T, sin_phi: T, (v_half, n, cos_n_phi0, sin_n_phi0): Self::Params) -> T {
        let one = T::from(1.0f32);
        let (cos_n_phi, sin_n_phi) = multiple_angle(cos_phi, sin_phi, n);
        let cos_n_delta = cos_n_phi * cos_n_phi0 + sin_n_phi * sin_n_phi0;
        v_half * (one - cos_n_delta)
    }

    /// Computes only the torque $T$.
    ///
    /// # Formula
    ///
    /// $$ T = V_{half} \cdot n \cdot \sin(n(\phi - \phi_0)) $$
    ///
    /// This factor allows computing forces via the chain rule:
    /// $$ \vec{F} = -T \cdot \nabla \phi $$
    #[inline(always)]
    fn diff(cos_phi: T, sin_phi: T, (v_half, n, cos_n_phi0, sin_n_phi0): Self::Params) -> T {
        let (cos_n_phi, sin_n_phi) = multiple_angle(cos_phi, sin_phi, n);
        let sin_n_delta = sin_n_phi * cos_n_phi0 - cos_n_phi * sin_n_phi0;
        let n_t = T::from(n as f32);
        v_half * n_t * sin_n_delta
    }

    /// Computes both energy and torque efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(
        cos_phi: T,
        sin_phi: T,
        (v_half, n, cos_n_phi0, sin_n_phi0): Self::Params,
    ) -> EnergyDiff<T> {
        let one = T::from(1.0f32);

        let (cos_n_phi, sin_n_phi) = multiple_angle(cos_phi, sin_phi, n);

        let cos_n_delta = cos_n_phi * cos_n_phi0 + sin_n_phi * sin_n_phi0;
        let sin_n_delta = sin_n_phi * cos_n_phi0 - cos_n_phi * sin_n_phi0;

        let energy = v_half * (one - cos_n_delta);

        let n_t = T::from(n as f32);
        let diff = v_half * n_t * sin_n_delta;

        EnergyDiff { energy, diff }
    }
}

/// Computes $(\cos(n\phi), \sin(n\phi))$ using optimized paths for common $n$.
#[inline(always)]
fn multiple_angle<T: Real>(cos_phi: T, sin_phi: T, n: u8) -> (T, T) {
    match n {
        0 => multiple_angle_0(),
        1 => (cos_phi, sin_phi),
        2 => multiple_angle_2(cos_phi, sin_phi),
        3 => multiple_angle_3(cos_phi, sin_phi),
        _ => multiple_angle_chebyshev(cos_phi, sin_phi, n),
    }
}

/// $n = 0$: $(\cos(0), \sin(0)) = (1, 0)$.
#[inline(always)]
fn multiple_angle_0<T: Real>() -> (T, T) {
    (T::from(1.0f32), T::from(0.0f32))
}

/// $n = 2$: Double-angle formulas.
#[inline(always)]
fn multiple_angle_2<T: Real>(cos_phi: T, sin_phi: T) -> (T, T) {
    let one = T::from(1.0f32);
    let two = T::from(2.0f32);

    let cos_2phi = two * cos_phi * cos_phi - one;
    let sin_2phi = two * sin_phi * cos_phi;

    (cos_2phi, sin_2phi)
}

/// $n = 3$: Triple-angle formulas.
#[inline(always)]
fn multiple_angle_3<T: Real>(cos_phi: T, sin_phi: T) -> (T, T) {
    let three = T::from(3.0f32);
    let four = T::from(4.0f32);

    let cos2 = cos_phi * cos_phi;
    let sin2 = sin_phi * sin_phi;

    let cos_3phi = four * cos2 * cos_phi - three * cos_phi;
    let sin_3phi = three * sin_phi - four * sin2 * sin_phi;

    (cos_3phi, sin_3phi)
}

/// General case: Chebyshev recurrence for $n \geq 4$.
#[inline(always)]
fn multiple_angle_chebyshev<T: Real>(cos_phi: T, sin_phi: T, n: u8) -> (T, T) {
    let zero = T::from(0.0f32);
    let one = T::from(1.0f32);
    let two = T::from(2.0f32);

    let mut cos_prev2 = one;
    let mut sin_prev2 = zero;
    let mut cos_prev1 = cos_phi;
    let mut sin_prev1 = sin_phi;

    let two_cos = two * cos_phi;

    for _ in 2..=n {
        let cos_curr = two_cos * cos_prev1 - cos_prev2;
        let sin_curr = two_cos * sin_prev1 - sin_prev2;

        cos_prev2 = cos_prev1;
        sin_prev2 = sin_prev1;
        cos_prev1 = cos_curr;
        sin_prev1 = sin_curr;
    }

    (cos_prev1, sin_prev1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    // ------------------------------------------------------------------------
    // Test Constants
    // ------------------------------------------------------------------------

    const H: f64 = 1e-6;
    const TOL_DIFF: f64 = 1e-4;

    // ========================================================================
    // Torsion Tests
    // ========================================================================

    mod torsion {
        use super::*;

        const V_HALF: f64 = 2.5; // V/2 = 2.5 kcal/mol

        fn params_n1() -> (f64, u8, f64, f64) {
            // n=1, phi0=0 => cos(0)=1, sin(0)=0
            (V_HALF, 1, 1.0, 0.0)
        }

        fn params_n2() -> (f64, u8, f64, f64) {
            // n=2, phi0=π => cos(2π)=1, sin(2π)=0
            (V_HALF, 2, 1.0, 0.0)
        }

        fn params_n3() -> (f64, u8, f64, f64) {
            // n=3, phi0=0 => cos(0)=1, sin(0)=0
            (V_HALF, 3, 1.0, 0.0)
        }

        fn params_n4() -> (f64, u8, f64, f64) {
            // n=4, phi0=0 => cos(0)=1, sin(0)=0
            (V_HALF, 4, 1.0, 0.0)
        }

        // --------------------------------------------------------------------
        // 1. Sanity Checks
        // --------------------------------------------------------------------

        #[test]
        fn sanity_compute_equals_separate() {
            let phi = PI / 4.0;
            let (cos_phi, sin_phi) = (phi.cos(), phi.sin());
            let p = params_n2();

            let result = Torsion::compute(cos_phi, sin_phi, p);
            let energy_only = Torsion::energy(cos_phi, sin_phi, p);
            let diff_only = Torsion::diff(cos_phi, sin_phi, p);

            assert_relative_eq!(result.energy, energy_only, epsilon = 1e-12);
            assert_relative_eq!(result.diff, diff_only, epsilon = 1e-12);
        }

        #[test]
        fn sanity_f32_f64_consistency() {
            let phi = PI / 3.0;
            let (cos_phi, sin_phi) = (phi.cos(), phi.sin());
            let p64 = params_n2();
            let p32 = (V_HALF as f32, 2u8, 1.0f32, 0.0f32);

            let e64 = Torsion::energy(cos_phi, sin_phi, p64);
            let e32 = Torsion::energy(cos_phi as f32, sin_phi as f32, p32);

            assert_relative_eq!(e64, e32 as f64, epsilon = 1e-5);
        }

        #[test]
        fn sanity_equilibrium_n2() {
            let (cos_phi, sin_phi) = (1.0_f64, 0.0_f64);
            let result = Torsion::compute(cos_phi, sin_phi, params_n2());

            assert_relative_eq!(result.energy, 0.0, epsilon = 1e-14);
            assert_relative_eq!(result.diff, 0.0, epsilon = 1e-14);
        }

        // --------------------------------------------------------------------
        // 2. Numerical Stability
        // --------------------------------------------------------------------

        #[test]
        fn stability_phi_zero() {
            let result = Torsion::compute(1.0, 0.0, params_n3());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
        }

        #[test]
        fn stability_phi_pi() {
            let result = Torsion::compute(-1.0, 0.0, params_n3());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
        }

        #[test]
        fn stability_high_periodicity() {
            let p = (V_HALF, 6u8, 1.0, 0.0);
            let phi = PI / 5.0;
            let result = Torsion::compute(phi.cos(), phi.sin(), p);

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
        }

        // --------------------------------------------------------------------
        // 3. Finite Difference Verification
        // --------------------------------------------------------------------

        fn finite_diff_check(phi: f64, params: (f64, u8, f64, f64)) {
            let (cos_phi, sin_phi) = (phi.cos(), phi.sin());

            let phi_plus = phi + H;
            let phi_minus = phi - H;
            let e_plus = Torsion::energy(phi_plus.cos(), phi_plus.sin(), params);
            let e_minus = Torsion::energy(phi_minus.cos(), phi_minus.sin(), params);
            let de_dphi_numerical = (e_plus - e_minus) / (2.0 * H);

            let torque = Torsion::diff(cos_phi, sin_phi, params);

            assert_relative_eq!(de_dphi_numerical, torque, epsilon = TOL_DIFF);
        }

        #[test]
        fn finite_diff_n1() {
            finite_diff_check(PI / 4.0, params_n1());
        }

        #[test]
        fn finite_diff_n2() {
            finite_diff_check(PI / 3.0, params_n2());
        }

        #[test]
        fn finite_diff_n3() {
            finite_diff_check(PI / 6.0, params_n3());
        }

        #[test]
        fn finite_diff_n4() {
            finite_diff_check(PI / 5.0, params_n4());
        }

        #[test]
        fn finite_diff_various_phi() {
            for phi in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0] {
                finite_diff_check(phi, params_n2());
            }
        }

        // --------------------------------------------------------------------
        // 4. Torsion Specific
        // --------------------------------------------------------------------

        #[test]
        fn specific_barrier_height() {
            let phi = PI;
            let e = Torsion::energy(phi.cos(), phi.sin(), params_n1());

            assert_relative_eq!(e, 2.0 * V_HALF, epsilon = 1e-10);
        }

        #[test]
        fn specific_n_minima() {
            let p = params_n3();
            let mut min_count = 0;
            for i in 0..36 {
                let phi = i as f64 * PI / 18.0;
                let e = Torsion::energy(phi.cos(), phi.sin(), p);
                if e < 0.01 {
                    min_count += 1;
                }
            }
            assert_eq!(min_count, 3);
        }

        #[test]
        fn specific_periodicity() {
            let phi = 0.5;
            let p = params_n3();
            let e1 = Torsion::energy(phi.cos(), phi.sin(), p);
            let phi2 = phi + 2.0 * PI / 3.0;
            let e2 = Torsion::energy(phi2.cos(), phi2.sin(), p);

            assert_relative_eq!(e1, e2, epsilon = 1e-10);
        }
    }

    // ========================================================================
    // Multiple Angle Helper Tests
    // ========================================================================

    mod multiple_angle_tests {
        use super::*;

        fn check_multiple_angle(phi: f64, n: u8) {
            let (cos_phi, sin_phi) = (phi.cos(), phi.sin());
            let (cos_n, sin_n) = multiple_angle(cos_phi, sin_phi, n);

            let expected_cos = (n as f64 * phi).cos();
            let expected_sin = (n as f64 * phi).sin();

            assert_relative_eq!(cos_n, expected_cos, epsilon = 1e-10);
            assert_relative_eq!(sin_n, expected_sin, epsilon = 1e-10);
        }

        #[test]
        fn multiple_angle_n0() {
            let (cos_n, sin_n) = multiple_angle(0.5_f64, 0.866, 0);
            assert_relative_eq!(cos_n, 1.0, epsilon = 1e-14);
            assert_relative_eq!(sin_n, 0.0, epsilon = 1e-14);
        }

        #[test]
        fn multiple_angle_n1() {
            let phi = PI / 4.0;
            check_multiple_angle(phi, 1);
        }

        #[test]
        fn multiple_angle_n2() {
            let phi = PI / 3.0;
            check_multiple_angle(phi, 2);
        }

        #[test]
        fn multiple_angle_n3() {
            let phi = PI / 5.0;
            check_multiple_angle(phi, 3);
        }

        #[test]
        fn multiple_angle_n4_chebyshev() {
            let phi = PI / 6.0;
            check_multiple_angle(phi, 4);
        }

        #[test]
        fn multiple_angle_n6_chebyshev() {
            let phi = PI / 7.0;
            check_multiple_angle(phi, 6);
        }
    }
}
