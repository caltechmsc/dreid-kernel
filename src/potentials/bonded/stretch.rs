use crate::math::Real;
use crate::traits::PairKernel;
use crate::types::EnergyDiff;

/// Harmonic potential implementation for 1-2 bond stretching.
///
/// # Physics
///
/// Models the bond stretching as a harmonic oscillator.
///
/// - **Formula**: $$ E = \frac{1}{2} K (R - R_0)^2 $$
/// - **Derivative Factor (`diff`)**: $$ D = -\frac{K (R - R_0)}{R} $$
///
/// # Parameters
///
/// - `k_half`: Half force constant $K_{half} = K/2$.
/// - `r0`: Equilibrium distance $R_0$.
///
/// # Inputs
///
/// - `r_sq`: Squared distance $r^2$ between two atoms.
///
/// # Implementation Notes
///
/// - Requires square root to obtain $R$.
/// - All intermediate calculations are shared between energy and force computations.
/// - Branchless and panic-free.
#[derive(Clone, Copy, Debug, Default)]
pub struct Harmonic;

impl<T: Real> PairKernel<T> for Harmonic {
    type Params = (T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = K_{half} (R - R_0)^2 $$
    #[inline(always)]
    fn energy(r_sq: T, (k_half, r0): Self::Params) -> T {
        let r = r_sq.sqrt();
        let delta = r - r0;
        k_half * delta * delta
    }

    /// Computes only the force pre-factor $D$.
    ///
    /// # Formula
    ///
    /// $$ D = -\frac{2 K_{half} (R - R_0)}{R} $$
    ///
    /// This factor is defined such that the force vector can be computed
    /// by a single vector multiplication: $\vec{F} = -D \cdot \vec{r}$.
    #[inline(always)]
    fn diff(r_sq: T, (k_half, r0): Self::Params) -> T {
        let inv_r = r_sq.rsqrt();
        let r = r_sq * inv_r;
        let delta = r - r0;

        let k = k_half + k_half;

        -k * delta * inv_r
    }

    /// Computes both energy and force pre-factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(r_sq: T, (k_half, r0): Self::Params) -> EnergyDiff<T> {
        let inv_r = r_sq.rsqrt();
        let r = r_sq * inv_r;
        let delta = r - r0;

        let energy = k_half * delta * delta;

        let k = k_half + k_half;
        let diff = -k * delta * inv_r;

        EnergyDiff { energy, diff }
    }
}

/// Morse potential implementation for 1-2 bond stretching.
///
/// # Physics
///
/// Models bond stretching with anharmonicity, allowing for bond dissociation.
///
/// - **Formula**: $$ E = D_e [ e^{-\alpha(R - R_0)} - 1 ]^2 $$
/// - **Derivative Factor (`diff`)**: $$ D = \frac{2 \alpha D_e e^{-\alpha(R - R_0)} \left( e^{-\alpha(R - R_0)} - 1 \right)}{R} $$
///
/// # Parameters
///
/// - `de`: Dissociation energy $D_e$.
/// - `r0`: Equilibrium distance $R_0$.
/// - `alpha`: Stiffness parameter $\alpha$.
///
/// # Inputs
///
/// - `r_sq`: Squared distance $r^2$ between two atoms.
///
/// # Implementation Notes
///
/// - Requires `sqrt` and `exp`.
/// - More computationally expensive than Harmonic.
#[derive(Clone, Copy, Debug, Default)]
pub struct Morse;

impl<T: Real> PairKernel<T> for Morse {
    type Params = (T, T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = D_e [ e^{-\alpha(R - R_0)} - 1 ]^2 $$
    #[inline(always)]
    fn energy(r_sq: T, (de, r0, alpha): Self::Params) -> T {
        let r = r_sq.sqrt();
        let t_val = T::exp(-alpha * (r - r0));
        let term = t_val - T::from(1.0);

        de * term * term
    }

    /// Computes only the force pre-factor $D$.
    ///
    /// # Formula
    ///
    /// $$ D = \frac{2 \alpha D_e e^{-\alpha(R - R_0)} \left( e^{-\alpha(R - R_0)} - 1 \right)}{R} $$
    ///
    /// This factor is defined such that the force vector can be computed
    /// by a single vector multiplication: $\vec{F} = -D \cdot \vec{r}$.
    #[inline(always)]
    fn diff(r_sq: T, (de, r0, alpha): Self::Params) -> T {
        let inv_r = r_sq.rsqrt();
        let r = r_sq * inv_r;

        let t_val = T::exp(-alpha * (r - r0));
        let term_minus_one = t_val - T::from(1.0);

        let f_mag = T::from(2.0) * alpha * de * t_val * term_minus_one;

        f_mag * inv_r
    }

    /// Computes both energy and force pre-factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(r_sq: T, (de, r0, alpha): Self::Params) -> EnergyDiff<T> {
        let inv_r = r_sq.rsqrt();
        let r = r_sq * inv_r;

        let t_val = T::exp(-alpha * (r - r0));
        let term_minus_one = t_val - T::from(1.0);

        let energy = de * term_minus_one * term_minus_one;

        let f_mag = T::from(2.0) * alpha * de * t_val * term_minus_one;
        let diff = f_mag * inv_r;

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
    const TOL_DIFF: f64 = 1e-4;

    // ========================================================================
    // Harmonic Tests
    // ========================================================================

    mod harmonic {
        use super::*;

        const K_HALF: f64 = 350.0; // K/2 = 350, so K = 700 kcal/(mol·Å²)
        const R0: f64 = 1.5; // Equilibrium distance Å

        fn params() -> (f64, f64) {
            (K_HALF, R0)
        }

        // --------------------------------------------------------------------
        // 1. Sanity Checks
        // --------------------------------------------------------------------

        #[test]
        fn sanity_compute_equals_separate() {
            let r_sq = 2.25_f64;
            let p = params();

            let result = Harmonic::compute(r_sq, p);
            let energy_only = Harmonic::energy(r_sq, p);
            let diff_only = Harmonic::diff(r_sq, p);

            assert_relative_eq!(result.energy, energy_only, epsilon = 1e-12);
            assert_relative_eq!(result.diff, diff_only, epsilon = 1e-12);
        }

        #[test]
        fn sanity_f32_f64_consistency() {
            let r_sq = 3.0;
            let p64 = params();
            let p32 = (K_HALF as f32, R0 as f32);

            let e64 = Harmonic::energy(r_sq, p64);
            let e32 = Harmonic::energy(r_sq as f32, p32);

            assert_relative_eq!(e64, e32 as f64, epsilon = 1e-4);
        }

        #[test]
        fn sanity_equilibrium() {
            let r0_sq = R0 * R0;
            let result = Harmonic::compute(r0_sq, params());

            assert_relative_eq!(result.energy, 0.0, epsilon = 1e-14);
            assert_relative_eq!(result.diff, 0.0, epsilon = 1e-14);
        }

        // --------------------------------------------------------------------
        // 2. Numerical Stability
        // --------------------------------------------------------------------

        #[test]
        fn stability_stretched() {
            let r = 3.0;
            let result = Harmonic::compute(r * r, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
            assert!(result.energy > 0.0);
        }

        #[test]
        fn stability_compressed() {
            let r = 0.5;
            let result = Harmonic::compute(r * r, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
            assert!(result.energy > 0.0);
        }

        // --------------------------------------------------------------------
        // 3. Finite Difference Verification
        // --------------------------------------------------------------------

        fn finite_diff_check(r: f64) {
            let p = params();
            let r_sq = r * r;

            let r_plus = r + H;
            let r_minus = r - H;
            let e_plus = Harmonic::energy(r_plus * r_plus, p);
            let e_minus = Harmonic::energy(r_minus * r_minus, p);
            let de_dr_numerical = (e_plus - e_minus) / (2.0 * H);

            let d = Harmonic::diff(r_sq, p);
            let de_dr_analytic = -d * r;

            assert_relative_eq!(de_dr_numerical, de_dr_analytic, epsilon = TOL_DIFF);
        }

        #[test]
        fn finite_diff_stretched() {
            finite_diff_check(2.0);
        }

        #[test]
        fn finite_diff_compressed() {
            finite_diff_check(1.0);
        }

        #[test]
        fn finite_diff_near_equilibrium() {
            finite_diff_check(R0 + 0.01);
        }

        // --------------------------------------------------------------------
        // 4. Harmonic Specific
        // --------------------------------------------------------------------

        #[test]
        fn specific_quadratic_scaling() {
            let delta1 = 0.1;
            let delta2 = 0.2;
            let e1 = Harmonic::energy((R0 + delta1).powi(2), params());
            let e2 = Harmonic::energy((R0 + delta2).powi(2), params());

            assert_relative_eq!(e2 / e1, 4.0, epsilon = 1e-10);
        }

        #[test]
        fn specific_force_restoring() {
            let d_stretched = Harmonic::diff((R0 + 0.5).powi(2), params());
            let d_compressed = Harmonic::diff((R0 - 0.5).powi(2), params());

            assert!(d_stretched < 0.0);
            assert!(d_compressed > 0.0);
        }
    }

    // ========================================================================
    // Morse Tests
    // ========================================================================

    mod morse {
        use super::*;

        const DE: f64 = 70.0; // Dissociation energy kcal/mol
        const R0: f64 = 1.5; // Equilibrium distance Å
        const ALPHA: f64 = 2.0; // Stiffness parameter Å⁻¹

        fn params() -> (f64, f64, f64) {
            (DE, R0, ALPHA)
        }

        // --------------------------------------------------------------------
        // 1. Sanity Checks
        // --------------------------------------------------------------------

        #[test]
        fn sanity_compute_equals_separate() {
            let r_sq = 2.0_f64;
            let p = params();

            let result = Morse::compute(r_sq, p);
            let energy_only = Morse::energy(r_sq, p);
            let diff_only = Morse::diff(r_sq, p);

            assert_relative_eq!(result.energy, energy_only, epsilon = 1e-12);
            assert_relative_eq!(result.diff, diff_only, epsilon = 1e-12);
        }

        #[test]
        fn sanity_f32_f64_consistency() {
            let r_sq = 3.0;
            let p64 = params();
            let p32 = (DE as f32, R0 as f32, ALPHA as f32);

            let e64 = Morse::energy(r_sq, p64);
            let e32 = Morse::energy(r_sq as f32, p32);

            assert_relative_eq!(e64, e32 as f64, epsilon = 1e-4);
        }

        #[test]
        fn sanity_equilibrium() {
            let r0_sq = R0 * R0;
            let result = Morse::compute(r0_sq, params());

            assert_relative_eq!(result.energy, 0.0, epsilon = 1e-14);
            assert_relative_eq!(result.diff, 0.0, epsilon = 1e-14);
        }

        // --------------------------------------------------------------------
        // 2. Numerical Stability
        // --------------------------------------------------------------------

        #[test]
        fn stability_large_stretch() {
            let r = 10.0;
            let result = Morse::compute(r * r, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
            assert_relative_eq!(result.energy, DE, epsilon = 1e-3);
        }

        #[test]
        fn stability_compressed() {
            let r = 0.5;
            let result = Morse::compute(r * r, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
            assert!(result.energy > DE);
        }

        // --------------------------------------------------------------------
        // 3. Finite Difference Verification
        // --------------------------------------------------------------------

        fn finite_diff_check(r: f64) {
            let p = params();
            let r_sq = r * r;

            let r_plus = r + H;
            let r_minus = r - H;
            let e_plus = Morse::energy(r_plus * r_plus, p);
            let e_minus = Morse::energy(r_minus * r_minus, p);
            let de_dr_numerical = (e_plus - e_minus) / (2.0 * H);

            let d = Morse::diff(r_sq, p);
            let de_dr_analytic = -d * r;

            assert_relative_eq!(de_dr_numerical, de_dr_analytic, epsilon = TOL_DIFF);
        }

        #[test]
        fn finite_diff_stretched() {
            finite_diff_check(2.5);
        }

        #[test]
        fn finite_diff_compressed() {
            finite_diff_check(1.0);
        }

        #[test]
        fn finite_diff_near_equilibrium() {
            finite_diff_check(R0 + 0.01);
        }

        // --------------------------------------------------------------------
        // 4. Morse Specific
        // --------------------------------------------------------------------

        #[test]
        fn specific_dissociation_limit() {
            let e_far = Morse::energy(100.0_f64.powi(2), params());
            assert_relative_eq!(e_far, DE, epsilon = 1e-6);
        }

        #[test]
        fn specific_anharmonic_asymmetry() {
            let delta = 0.3;
            let e_stretch = Morse::energy((R0 + delta).powi(2), params());
            let e_compress = Morse::energy((R0 - delta).powi(2), params());

            assert!(e_compress > e_stretch);
        }

        #[test]
        fn specific_force_restoring() {
            let d_stretched = Morse::diff((R0 + 0.5).powi(2), params());
            let d_compressed = Morse::diff((R0 - 0.3).powi(2), params());

            assert!(d_stretched < 0.0);
            assert!(d_compressed > 0.0);
        }
    }
}
