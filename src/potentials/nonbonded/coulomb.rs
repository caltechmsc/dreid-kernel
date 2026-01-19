use crate::math::Real;
use crate::traits::PairKernel;
use crate::types::EnergyDiff;

/// Standard Coulomb potential implementation for electrostatics.
///
/// # Physics
///
/// Models the electrostatic interaction between two point charges.
///
/// - **Formula**: $$ E = \frac{C \cdot q_i q_j}{\epsilon \cdot r} $$
/// - **Derivative Factor (`diff`)**: $$ D = -\frac{1}{r} \frac{dE}{dr} = \frac{E}{r^2} = \frac{C \cdot q_i q_j}{\epsilon \cdot r^3} $$
///
/// # Parameters
///
/// - `q_product`: The effective charge product $ Q_{eff} = \frac{C \cdot q_i q_j}{\epsilon} $.
///
/// # Inputs
///
/// - `r_sq`: Squared distance $ r^2 $ between two atoms.
///
/// # Implementation Notes
///
/// - Optimized to minimize arithmetic operations by reusing intermediate `inv_r` and `inv_r2`.
/// - Utilizes `rsqrt` for potential hardware acceleration.
/// - Branchless and panic-free.
#[derive(Clone, Copy, Debug, Default)]
pub struct Coulomb;

impl<T: Real> PairKernel<T> for Coulomb {
    type Params = T;

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = \frac{Q_{eff}}{r} $$
    #[inline(always)]
    fn energy(r_sq: T, q_product: Self::Params) -> T {
        q_product * r_sq.rsqrt()
    }

    /// Computes only the force pre-factor $D$.
    ///
    /// # Formula
    ///
    /// $$ D = \frac{Q_{eff}}{r^3} $$
    ///
    /// This factor is defined such that the force vector can be computed
    /// by a single vector multiplication: $\vec{F} = -D \cdot \vec{r}$.
    #[inline(always)]
    fn diff(r_sq: T, q_product: Self::Params) -> T {
        let inv_r = r_sq.rsqrt();
        let energy = q_product * inv_r;

        let inv_r2 = inv_r * inv_r;
        energy * inv_r2
    }

    /// Computes both energy and force pre-factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(r_sq: T, q_product: Self::Params) -> EnergyDiff<T> {
        let inv_r = r_sq.rsqrt();
        let inv_r2 = inv_r * inv_r;

        let energy = q_product * inv_r;

        let diff = energy * inv_r2;

        EnergyDiff { energy, diff }
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
    const TOL_DIFF: f64 = 1e-5;

    // Typical charge product: Q_eff = 332.0637 * q_i * q_j / epsilon
    // For testing: Q_eff = 1.0 (simplified)
    const Q_EFF: f64 = 1.0;

    // ------------------------------------------------------------------------
    // 1. Sanity Checks
    // ------------------------------------------------------------------------

    #[test]
    fn sanity_compute_equals_separate() {
        let r_sq = 4.0_f64;
        let params = Q_EFF;

        let result = Coulomb::compute(r_sq, params);
        let energy_only = Coulomb::energy(r_sq, params);
        let diff_only = Coulomb::diff(r_sq, params);

        assert_relative_eq!(result.energy, energy_only, epsilon = 1e-14);
        assert_relative_eq!(result.diff, diff_only, epsilon = 1e-14);
    }

    #[test]
    fn sanity_f32_f64_consistency() {
        let r_sq_64 = 4.0_f64;
        let r_sq_32 = 4.0_f32;

        let e64 = Coulomb::energy(r_sq_64, Q_EFF);
        let e32 = Coulomb::energy(r_sq_32, Q_EFF as f32);

        assert_relative_eq!(e64, e32 as f64, epsilon = 1e-5);
    }

    #[test]
    fn sanity_energy_positive_for_like_charges() {
        let r_sq = 4.0_f64;
        let e = Coulomb::energy(r_sq, 1.0);
        assert!(e > 0.0);
    }

    #[test]
    fn sanity_energy_negative_for_unlike_charges() {
        let r_sq = 4.0_f64;
        let e: f64 = Coulomb::energy(r_sq, -1.0);
        assert!(e < 0.0);
    }

    // ------------------------------------------------------------------------
    // 2. Numerical Stability
    // ------------------------------------------------------------------------

    #[test]
    fn stability_large_distance() {
        let r_sq = 1e10_f64;
        let result = Coulomb::compute(r_sq, Q_EFF);

        assert!(result.energy.is_finite());
        assert!(result.diff.is_finite());
        assert!(result.energy.abs() < 1e-4);
        assert!(result.diff.abs() < 1e-14);
    }

    #[test]
    fn stability_small_distance() {
        let r_sq = 1e-6_f64;
        let result = Coulomb::compute(r_sq, Q_EFF);

        assert!(result.energy.is_finite());
        assert!(result.diff.is_finite());
        assert!(result.energy > 0.0);
    }

    // ------------------------------------------------------------------------
    // 3. Finite Difference Verification
    // ------------------------------------------------------------------------

    fn finite_diff_check(r: f64, params: f64) {
        let r_sq = r * r;

        let r_plus = r + H;
        let r_minus = r - H;
        let e_plus = Coulomb::energy(r_plus * r_plus, params);
        let e_minus = Coulomb::energy(r_minus * r_minus, params);
        let de_dr_numerical = (e_plus - e_minus) / (2.0 * H);

        let d_analytic = Coulomb::diff(r_sq, params);
        let de_dr_analytic = -d_analytic * r;

        assert_relative_eq!(de_dr_numerical, de_dr_analytic, epsilon = TOL_DIFF);
    }

    #[test]
    fn finite_diff_short_range() {
        finite_diff_check(1.0, Q_EFF);
    }

    #[test]
    fn finite_diff_medium_range() {
        finite_diff_check(3.0, Q_EFF);
    }

    #[test]
    fn finite_diff_long_range() {
        finite_diff_check(10.0, Q_EFF);
    }

    #[test]
    fn finite_diff_negative_charge() {
        finite_diff_check(2.5, -Q_EFF);
    }

    // ------------------------------------------------------------------------
    // 4. Coulomb-Specific: Scaling Laws
    // ------------------------------------------------------------------------

    #[test]
    fn specific_inverse_r_scaling() {
        let e1 = Coulomb::energy(1.0, Q_EFF);
        let e2 = Coulomb::energy(4.0, Q_EFF);

        assert_relative_eq!(e1 / e2, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn specific_inverse_r3_scaling_for_diff() {
        let d1 = Coulomb::diff(1.0, Q_EFF);
        let d2 = Coulomb::diff(4.0, Q_EFF);

        assert_relative_eq!(d1 / d2, 8.0, epsilon = 1e-10);
    }
}
