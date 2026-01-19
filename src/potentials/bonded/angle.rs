use crate::math::Real;
use crate::traits::AngleKernel;
use crate::types::EnergyDiff;

/// Cosine Harmonic potential for bond angles.
///
/// # Physics
///
/// Models the angle bending energy using a simple harmonic approximation on the cosine of the angle.
///
/// - **Formula**: $$ E = \frac{1}{2} C (\cos\theta - \cos\theta_0)^2 $$
/// - **Derivative Factor (`diff`)**: $$ \Gamma = \frac{dE}{d(\cos\theta)} = C (\cos\theta - \cos\theta_0) $$
///
/// # Parameters
///
/// - `k_half`: Half force constant $C_{half} = C/2$.
/// - `cos0`: The cosine of the equilibrium angle $\cos\theta_0$.
///
/// # Inputs
///
/// - `cos_theta`: Cosine of the bond angle $\theta_{ijk}$.
///
/// # Implementation Notes
///
/// - Pure polynomial evaluation; no `acos`, `sin`, or `sqrt` required.
/// - All intermediate calculations are shared between energy and force computations.
/// - Branchless and panic-free.
#[derive(Clone, Copy, Debug, Default)]
pub struct CosineHarmonic;

impl<T: Real> AngleKernel<T> for CosineHarmonic {
    type Params = (T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = C_{half} (\Delta)^2, \quad \text{where } \Delta = \cos\theta - \cos\theta_0 $$
    #[inline(always)]
    fn energy(cos_theta: T, (k_half, cos0): Self::Params) -> T {
        let delta = cos_theta - cos0;
        k_half * delta * delta
    }

    /// Computes only the derivative factor $\Gamma$.
    ///
    /// # Formula
    ///
    /// $$ \Gamma = 2 C_{half} (\cos\theta - \cos\theta_0) $$
    ///
    /// This factor allows computing forces via the chain rule:
    /// $$ \vec{F} = -\Gamma \cdot \nabla (\cos\theta) $$
    #[inline(always)]
    fn diff(cos_theta: T, (k_half, cos0): Self::Params) -> T {
        let c = k_half + k_half;
        c * (cos_theta - cos0)
    }

    /// Computes both energy and derivative factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(cos_theta: T, (k_half, cos0): Self::Params) -> EnergyDiff<T> {
        let delta = cos_theta - cos0;

        let energy = k_half * delta * delta;

        let c = k_half + k_half;
        let diff = c * delta;

        EnergyDiff { energy, diff }
    }
}

/// Theta Harmonic potential for bond angles.
///
/// # Physics
///
/// Models the angle bending energy using a harmonic approximation directly on the angle $\theta$ (in radians).
///
/// - **Formula**: $$ E = \frac{1}{2} K (\theta - \theta_0)^2 $$
/// - **Derivative Factor (`diff`)**: $$ \Gamma = \frac{dE}{d(\cos\theta)} = -K \frac{\theta - \theta_0}{\sin\theta} $$
///
/// # Parameters
///
/// - `k_half`: Half force constant $K_{half} = K/2$.
/// - `theta0`: The equilibrium angle $\theta_0$ in radians.
///
/// # Inputs
///
/// - `cos_theta`: Cosine of the bond angle $\theta_{ijk}$.
///
/// # Implementation Notes
///
/// - Uses `k_half` to save one multiplication in the energy step.
/// - Handles $\theta=0$ and $\theta=\pi$ analytically using L'Hopital's rule.
/// - Needs a single `acos` call for angle calculation.
#[derive(Clone, Copy, Debug, Default)]
pub struct ThetaHarmonic;

impl<T: Real> AngleKernel<T> for ThetaHarmonic {
    type Params = (T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = K_{half} (\theta - \theta_0)^2 $$
    #[inline(always)]
    fn energy(cos_theta: T, (k_half, theta0): Self::Params) -> T {
        let one = T::from(1.0f32);
        let minus_one = T::from(-1.0f32);

        let c = cos_theta.max(minus_one).min(one);
        let theta = c.acos();

        let delta = theta - theta0;

        k_half * delta * delta
    }

    /// Computes only the derivative factor $\Gamma$.
    ///
    /// # Formula
    ///
    /// $$ \Gamma = -2 K_{half} \frac{\theta - \theta_0}{\sin\theta} $$
    ///
    /// This factor allows computing forces via the chain rule:
    /// $$ \vec{F} = -\Gamma \cdot \nabla (\cos\theta) $$
    #[inline(always)]
    fn diff(cos_theta: T, (k_half, theta0): Self::Params) -> T {
        let one = T::from(1.0f32);
        let minus_one = T::from(-1.0f32);
        let zero = T::from(0.0f32);
        let singularity_thresh = T::from(1.0e-4f32);
        let epsilon = T::from(1.0e-20f32);

        let c = cos_theta.max(minus_one).min(one);

        let theta = c.acos();
        let sin_theta = (one - c * c).max(zero).sqrt();

        let factor = if sin_theta > singularity_thresh {
            (theta - theta0) / sin_theta
        } else {
            let s_safe = sin_theta.max(epsilon);

            if c > zero {
                one - theta0 / s_safe
            } else {
                let pi = T::pi();
                minus_one + (pi - theta0) / s_safe
            }
        };

        let k = k_half + k_half;

        -k * factor
    }

    /// Computes both energy and derivative factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(cos_theta: T, (k_half, theta0): Self::Params) -> EnergyDiff<T> {
        let one = T::from(1.0f32);
        let minus_one = T::from(-1.0f32);
        let zero = T::from(0.0f32);
        let singularity_thresh = T::from(1.0e-4f32);
        let epsilon = T::from(1.0e-20f32);

        let c = cos_theta.max(minus_one).min(one);
        let theta = c.acos();
        let sin_theta = (one - c * c).max(zero).sqrt();

        let delta = theta - theta0;
        let energy = k_half * delta * delta;

        let factor = if sin_theta > singularity_thresh {
            delta / sin_theta
        } else {
            let s_safe = sin_theta.max(epsilon);
            if c > zero {
                one - theta0 / s_safe
            } else {
                let pi = T::pi();
                minus_one + (pi - theta0) / s_safe
            }
        };

        let k = k_half + k_half;
        let diff = -k * factor;

        EnergyDiff { energy, diff }
    }
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
    // CosineHarmonic Tests
    // ========================================================================

    mod cosine_harmonic {
        use super::*;

        const K_HALF: f64 = 50.0; // C/2 = 50
        const COS0: f64 = 0.5; // cos(60°) = 0.5

        fn params() -> (f64, f64) {
            (K_HALF, COS0)
        }

        // --------------------------------------------------------------------
        // 1. Sanity Checks
        // --------------------------------------------------------------------

        #[test]
        fn sanity_compute_equals_separate() {
            let cos_theta = 0.7_f64;
            let p = params();

            let result = CosineHarmonic::compute(cos_theta, p);
            let energy_only = CosineHarmonic::energy(cos_theta, p);
            let diff_only = CosineHarmonic::diff(cos_theta, p);

            assert_relative_eq!(result.energy, energy_only, epsilon = 1e-12);
            assert_relative_eq!(result.diff, diff_only, epsilon = 1e-12);
        }

        #[test]
        fn sanity_f32_f64_consistency() {
            let cos_theta = 0.6;
            let p64 = params();
            let p32 = (K_HALF as f32, COS0 as f32);

            let e64 = CosineHarmonic::energy(cos_theta, p64);
            let e32 = CosineHarmonic::energy(cos_theta as f32, p32);

            assert_relative_eq!(e64, e32 as f64, epsilon = 1e-5);
        }

        #[test]
        fn sanity_equilibrium() {
            let result = CosineHarmonic::compute(COS0, params());

            assert_relative_eq!(result.energy, 0.0, epsilon = 1e-14);
            assert_relative_eq!(result.diff, 0.0, epsilon = 1e-14);
        }

        // --------------------------------------------------------------------
        // 2. Numerical Stability
        // --------------------------------------------------------------------

        #[test]
        fn stability_cos_one() {
            let result = CosineHarmonic::compute(1.0, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
        }

        #[test]
        fn stability_cos_minus_one() {
            let result = CosineHarmonic::compute(-1.0, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
        }

        // --------------------------------------------------------------------
        // 3. Finite Difference Verification
        // --------------------------------------------------------------------

        fn finite_diff_check(cos_theta: f64) {
            let p = params();

            let c_plus = cos_theta + H;
            let c_minus = cos_theta - H;
            let e_plus = CosineHarmonic::energy(c_plus, p);
            let e_minus = CosineHarmonic::energy(c_minus, p);
            let de_dcos_numerical = (e_plus - e_minus) / (2.0 * H);

            let gamma = CosineHarmonic::diff(cos_theta, p);

            assert_relative_eq!(de_dcos_numerical, gamma, epsilon = TOL_DIFF);
        }

        #[test]
        fn finite_diff_bent() {
            finite_diff_check(0.3);
        }

        #[test]
        fn finite_diff_wide() {
            finite_diff_check(0.8);
        }

        #[test]
        fn finite_diff_near_equilibrium() {
            finite_diff_check(COS0 + 0.01);
        }

        // --------------------------------------------------------------------
        // 4. CosineHarmonic Specific
        // --------------------------------------------------------------------

        #[test]
        fn specific_quadratic_scaling() {
            let delta1 = 0.1;
            let delta2 = 0.2;
            let e1 = CosineHarmonic::energy(COS0 + delta1, params());
            let e2 = CosineHarmonic::energy(COS0 + delta2, params());

            assert_relative_eq!(e2 / e1, 4.0, epsilon = 1e-10);
        }

        #[test]
        fn specific_no_trig_needed() {
            let cos_theta = 0.7;
            let delta = cos_theta - COS0;
            let expected = K_HALF * delta * delta;

            assert_relative_eq!(
                CosineHarmonic::energy(cos_theta, params()),
                expected,
                epsilon = 1e-14
            );
        }
    }

    // ========================================================================
    // ThetaHarmonic Tests
    // ========================================================================

    mod theta_harmonic {
        use super::*;

        const K_HALF: f64 = 50.0; // K/2 = 50
        const THETA0: f64 = PI / 3.0; // 60° equilibrium
        const COS0: f64 = 0.5; // cos(60°)

        fn params() -> (f64, f64) {
            (K_HALF, THETA0)
        }

        // --------------------------------------------------------------------
        // 1. Sanity Checks
        // --------------------------------------------------------------------

        #[test]
        fn sanity_compute_equals_separate() {
            let cos_theta = 0.7_f64;
            let p = params();

            let result = ThetaHarmonic::compute(cos_theta, p);
            let energy_only = ThetaHarmonic::energy(cos_theta, p);
            let diff_only = ThetaHarmonic::diff(cos_theta, p);

            assert_relative_eq!(result.energy, energy_only, epsilon = 1e-12);
            assert_relative_eq!(result.diff, diff_only, epsilon = 1e-12);
        }

        #[test]
        fn sanity_f32_f64_consistency() {
            let cos_theta = 0.6;
            let p64 = params();
            let p32 = (K_HALF as f32, THETA0 as f32);

            let e64 = ThetaHarmonic::energy(cos_theta, p64);
            let e32 = ThetaHarmonic::energy(cos_theta as f32, p32);

            assert_relative_eq!(e64, e32 as f64, epsilon = 1e-4);
        }

        #[test]
        fn sanity_equilibrium() {
            let result = ThetaHarmonic::compute(COS0, params());

            assert_relative_eq!(result.energy, 0.0, epsilon = 1e-10);
            assert!(result.diff.abs() < 1e-8);
        }

        // --------------------------------------------------------------------
        // 2. Numerical Stability
        // --------------------------------------------------------------------

        #[test]
        fn stability_theta_zero() {
            let result = ThetaHarmonic::compute(1.0, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
        }

        #[test]
        fn stability_theta_pi() {
            let result = ThetaHarmonic::compute(-1.0, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
        }

        #[test]
        fn stability_cos_slightly_outside() {
            let result = ThetaHarmonic::compute(1.0001, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
        }

        // --------------------------------------------------------------------
        // 3. Finite Difference Verification
        // --------------------------------------------------------------------

        fn finite_diff_check(cos_theta: f64) {
            if cos_theta.abs() > 0.999 {
                return;
            }

            let p = params();

            let c_plus = cos_theta + H;
            let c_minus = cos_theta - H;
            let e_plus = ThetaHarmonic::energy(c_plus, p);
            let e_minus = ThetaHarmonic::energy(c_minus, p);
            let de_dcos_numerical = (e_plus - e_minus) / (2.0 * H);

            let gamma = ThetaHarmonic::diff(cos_theta, p);

            assert_relative_eq!(de_dcos_numerical, gamma, epsilon = TOL_DIFF);
        }

        #[test]
        fn finite_diff_acute() {
            finite_diff_check(0.7);
        }

        #[test]
        fn finite_diff_right() {
            finite_diff_check(0.0);
        }

        #[test]
        fn finite_diff_obtuse() {
            finite_diff_check(-0.5);
        }

        #[test]
        fn finite_diff_near_equilibrium() {
            finite_diff_check(COS0 + 0.02);
        }

        // --------------------------------------------------------------------
        // 4. ThetaHarmonic Specific
        // --------------------------------------------------------------------

        #[test]
        fn specific_energy_formula() {
            let cos_theta = 0.0;
            let theta = PI / 2.0;
            let delta = theta - THETA0;
            let expected = K_HALF * delta * delta;

            assert_relative_eq!(
                ThetaHarmonic::energy(cos_theta, params()),
                expected,
                epsilon = 1e-10
            );
        }

        #[test]
        fn specific_diff_chain_rule() {
            let cos_theta = 0.0;
            let theta = PI / 2.0;
            let sin_theta = 1.0;

            let k = 2.0 * K_HALF;
            let expected_gamma = -k * (theta - THETA0) / sin_theta;

            assert_relative_eq!(
                ThetaHarmonic::diff(cos_theta, params()),
                expected_gamma,
                epsilon = 1e-10
            );
        }
    }
}
