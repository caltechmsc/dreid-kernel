use crate::math::Real;
use crate::traits::AngleKernel;
use crate::types::EnergyDiff;

/// Planar inversion potential for sp² centers.
///
/// # Physics
///
/// Models planarity for sp² hybridized atoms (e.g., aromatic carbons, carbonyl groups).
/// The central atom $I$ is bonded to three neighbors $J, K, L$, and the potential
/// penalizes any out-of-plane displacement.
///
/// - **Formula**: $$ E = \frac{1}{2} C \cos^2\psi $$
/// - **Derivative Factor (`diff`)**: $$ \Gamma = \frac{dE}{d(\cos\psi)} = C \cos\psi $$
///
/// # Parameters
///
/// - `c_half`: Half force constant $C_{half} = C/2$.
///
/// # Inputs
///
/// - `cos_psi`: Cosine of the out-of-plane angle $\psi$.
///
/// # Implementation Notes
///
/// - Optimized for $\cos\psi_0 = 0$: avoids one subtraction per evaluation.
/// - Pure polynomial evaluation; no trigonometric functions required.
/// - All intermediate calculations are shared between energy and force computations.
/// - Branchless and panic-free.
/// - **DREIDING convention:** force constant $C = K_{inv}/3$ (split among three permutations).
#[derive(Clone, Copy, Debug, Default)]
pub struct PlanarInversion;

impl<T: Real> AngleKernel<T> for PlanarInversion {
    type Params = T;

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = C_{half} \cos^2\psi $$
    #[inline(always)]
    fn energy(cos_psi: T, c_half: Self::Params) -> T {
        c_half * cos_psi * cos_psi
    }

    /// Computes only the derivative factor $\Gamma$.
    ///
    /// # Formula
    ///
    /// $$ \Gamma = 2 C_{half} \cos\psi $$
    ///
    /// This factor allows computing forces via the chain rule:
    /// $$ \vec{F} = -\Gamma \cdot \nabla (\cos\psi) $$
    #[inline(always)]
    fn diff(cos_psi: T, c_half: Self::Params) -> T {
        let c = c_half + c_half;
        c * cos_psi
    }

    /// Computes both energy and derivative factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(cos_psi: T, c_half: Self::Params) -> EnergyDiff<T> {
        let energy = c_half * cos_psi * cos_psi;

        let c = c_half + c_half;
        let diff = c * cos_psi;

        EnergyDiff { energy, diff }
    }
}

/// Umbrella inversion potential for sp³ centers.
///
/// # Physics
///
/// Models a specific pyramidal geometry for sp³ hybridized atoms (e.g., amines, phosphines).
/// The central atom $I$ is bonded to three neighbors $J, K, L$, and the potential
/// penalizes deviations from the target out-of-plane angle $\psi_0$.
///
/// - **Formula**: $$ E = \frac{1}{2} C (\cos\psi - \cos\psi_0)^2 $$
/// - **Derivative Factor (`diff`)**: $$ \Gamma = \frac{dE}{d(\cos\psi)} = C (\cos\psi - \cos\psi_0) $$
///
/// # Parameters
///
/// - `c_half`: Half force constant $C_{half} = C/2$.
/// - `cos_psi0`: Equilibrium value $\cos\psi_0 \neq 0$.
///
/// # Inputs
///
/// - `cos_psi`: Cosine of the out-of-plane angle $\psi$.
///
/// # Implementation Notes
///
/// - Pure polynomial evaluation; no trigonometric functions required.
/// - All intermediate calculations are shared between energy and force computations.
/// - Branchless and panic-free.
/// - **DREIDING convention:** force constant $C = K_{inv}/3$ (split among three permutations).
#[derive(Clone, Copy, Debug, Default)]
pub struct UmbrellaInversion;

impl<T: Real> AngleKernel<T> for UmbrellaInversion {
    type Params = (T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = C_{half} (\cos\psi - \cos\psi_0)^2 $$
    #[inline(always)]
    fn energy(cos_psi: T, (c_half, cos_psi0): Self::Params) -> T {
        let delta = cos_psi - cos_psi0;
        c_half * delta * delta
    }

    /// Computes only the derivative factor $\Gamma$.
    ///
    /// # Formula
    ///
    /// $$ \Gamma = 2 C_{half} (\cos\psi - \cos\psi_0) $$
    ///
    /// This factor allows computing forces via the chain rule:
    /// $$ \vec{F} = -\Gamma \cdot \nabla (\cos\psi) $$
    #[inline(always)]
    fn diff(cos_psi: T, (c_half, cos_psi0): Self::Params) -> T {
        let c = c_half + c_half;
        c * (cos_psi - cos_psi0)
    }

    /// Computes both energy and derivative factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(cos_psi: T, (c_half, cos_psi0): Self::Params) -> EnergyDiff<T> {
        let delta = cos_psi - cos_psi0;

        let energy = c_half * delta * delta;

        let c = c_half + c_half;
        let diff = c * delta;

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
    // PlanarInversion Tests
    // ========================================================================

    mod planar_inversion {
        use super::*;

        const C_HALF: f64 = 20.0; // C/2 = 20 kcal/mol

        fn params() -> f64 {
            C_HALF
        }

        // --------------------------------------------------------------------
        // 1. Sanity Checks
        // --------------------------------------------------------------------

        #[test]
        fn sanity_compute_equals_separate() {
            let cos_psi = 0.3_f64;
            let p = params();

            let result = PlanarInversion::compute(cos_psi, p);
            let energy_only = PlanarInversion::energy(cos_psi, p);
            let diff_only = PlanarInversion::diff(cos_psi, p);

            assert_relative_eq!(result.energy, energy_only, epsilon = 1e-12);
            assert_relative_eq!(result.diff, diff_only, epsilon = 1e-12);
        }

        #[test]
        fn sanity_f32_f64_consistency() {
            let cos_psi = 0.4;
            let p64 = params();
            let p32 = C_HALF as f32;

            let e64 = PlanarInversion::energy(cos_psi, p64);
            let e32 = PlanarInversion::energy(cos_psi as f32, p32);

            assert_relative_eq!(e64, e32 as f64, epsilon = 1e-5);
        }

        #[test]
        fn sanity_equilibrium_planar() {
            let result = PlanarInversion::compute(0.0, params());

            assert_relative_eq!(result.energy, 0.0, epsilon = 1e-14);
            assert_relative_eq!(result.diff, 0.0, epsilon = 1e-14);
        }

        // --------------------------------------------------------------------
        // 2. Numerical Stability
        // --------------------------------------------------------------------

        #[test]
        fn stability_cos_one() {
            let result = PlanarInversion::compute(1.0, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
            assert_relative_eq!(result.energy, C_HALF, epsilon = 1e-10);
        }

        #[test]
        fn stability_cos_minus_one() {
            let result = PlanarInversion::compute(-1.0, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
            assert_relative_eq!(result.energy, C_HALF, epsilon = 1e-10);
        }

        // --------------------------------------------------------------------
        // 3. Finite Difference Verification
        // --------------------------------------------------------------------

        fn finite_diff_check(cos_psi: f64) {
            let p = params();

            let c_plus = cos_psi + H;
            let c_minus = cos_psi - H;
            let e_plus = PlanarInversion::energy(c_plus, p);
            let e_minus = PlanarInversion::energy(c_minus, p);
            let de_dcos_numerical = (e_plus - e_minus) / (2.0 * H);

            let gamma = PlanarInversion::diff(cos_psi, p);

            assert_relative_eq!(de_dcos_numerical, gamma, epsilon = TOL_DIFF);
        }

        #[test]
        fn finite_diff_small_deviation() {
            finite_diff_check(0.1);
        }

        #[test]
        fn finite_diff_medium_deviation() {
            finite_diff_check(0.5);
        }

        #[test]
        fn finite_diff_large_deviation() {
            finite_diff_check(0.9);
        }

        #[test]
        fn finite_diff_negative() {
            finite_diff_check(-0.3);
        }

        // --------------------------------------------------------------------
        // 4. PlanarInversion Specific
        // --------------------------------------------------------------------

        #[test]
        fn specific_quadratic_in_cos() {
            let cos_psi = 0.6;
            let expected = C_HALF * cos_psi * cos_psi;

            assert_relative_eq!(
                PlanarInversion::energy(cos_psi, params()),
                expected,
                epsilon = 1e-14
            );
        }

        #[test]
        fn specific_symmetric_deviation() {
            let e_pos = PlanarInversion::energy(0.5, params());
            let e_neg = PlanarInversion::energy(-0.5, params());

            assert_relative_eq!(e_pos, e_neg, epsilon = 1e-14);
        }

        #[test]
        fn specific_restoring_force_direction() {
            let d_pos = PlanarInversion::diff(0.5, params());
            let d_neg = PlanarInversion::diff(-0.5, params());

            assert!(d_pos > 0.0);
            assert!(d_neg < 0.0);
        }
    }

    // ========================================================================
    // UmbrellaInversion Tests
    // ========================================================================

    mod umbrella_inversion {
        use super::*;

        const C_HALF: f64 = 15.0; // C/2 = 15 kcal/mol
        const COS_PSI0: f64 = 0.33; // ~70° typical for sp³

        fn params() -> (f64, f64) {
            (C_HALF, COS_PSI0)
        }

        // --------------------------------------------------------------------
        // 1. Sanity Checks
        // --------------------------------------------------------------------

        #[test]
        fn sanity_compute_equals_separate() {
            let cos_psi = 0.5_f64;
            let p = params();

            let result = UmbrellaInversion::compute(cos_psi, p);
            let energy_only = UmbrellaInversion::energy(cos_psi, p);
            let diff_only = UmbrellaInversion::diff(cos_psi, p);

            assert_relative_eq!(result.energy, energy_only, epsilon = 1e-12);
            assert_relative_eq!(result.diff, diff_only, epsilon = 1e-12);
        }

        #[test]
        fn sanity_f32_f64_consistency() {
            let cos_psi = 0.4;
            let p64 = params();
            let p32 = (C_HALF as f32, COS_PSI0 as f32);

            let e64 = UmbrellaInversion::energy(cos_psi, p64);
            let e32 = UmbrellaInversion::energy(cos_psi as f32, p32);

            assert_relative_eq!(e64, e32 as f64, epsilon = 1e-5);
        }

        #[test]
        fn sanity_equilibrium() {
            let result = UmbrellaInversion::compute(COS_PSI0, params());

            assert_relative_eq!(result.energy, 0.0, epsilon = 1e-14);
            assert_relative_eq!(result.diff, 0.0, epsilon = 1e-14);
        }

        // --------------------------------------------------------------------
        // 2. Numerical Stability
        // --------------------------------------------------------------------

        #[test]
        fn stability_cos_one() {
            let result = UmbrellaInversion::compute(1.0, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
        }

        #[test]
        fn stability_cos_minus_one() {
            let result = UmbrellaInversion::compute(-1.0, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
        }

        // --------------------------------------------------------------------
        // 3. Finite Difference Verification
        // --------------------------------------------------------------------

        fn finite_diff_check(cos_psi: f64) {
            let p = params();

            let c_plus = cos_psi + H;
            let c_minus = cos_psi - H;
            let e_plus = UmbrellaInversion::energy(c_plus, p);
            let e_minus = UmbrellaInversion::energy(c_minus, p);
            let de_dcos_numerical = (e_plus - e_minus) / (2.0 * H);

            let gamma = UmbrellaInversion::diff(cos_psi, p);

            assert_relative_eq!(de_dcos_numerical, gamma, epsilon = TOL_DIFF);
        }

        #[test]
        fn finite_diff_above_equilibrium() {
            finite_diff_check(COS_PSI0 + 0.2);
        }

        #[test]
        fn finite_diff_below_equilibrium() {
            finite_diff_check(COS_PSI0 - 0.2);
        }

        #[test]
        fn finite_diff_at_zero() {
            finite_diff_check(0.0);
        }

        #[test]
        fn finite_diff_negative() {
            finite_diff_check(-0.5);
        }

        // --------------------------------------------------------------------
        // 4. UmbrellaInversion Specific
        // --------------------------------------------------------------------

        #[test]
        fn specific_quadratic_in_delta() {
            let cos_psi = 0.6;
            let delta = cos_psi - COS_PSI0;
            let expected = C_HALF * delta * delta;

            assert_relative_eq!(
                UmbrellaInversion::energy(cos_psi, params()),
                expected,
                epsilon = 1e-14
            );
        }

        #[test]
        fn specific_reduces_to_planar_when_cos0_zero() {
            let p_umbrella = (C_HALF, 0.0);
            let p_planar = C_HALF;
            let cos_psi = 0.5;

            let e_umbrella = UmbrellaInversion::energy(cos_psi, p_umbrella);
            let e_planar = PlanarInversion::energy(cos_psi, p_planar);

            assert_relative_eq!(e_umbrella, e_planar, epsilon = 1e-14);
        }

        #[test]
        fn specific_restoring_force_direction() {
            let d_above = UmbrellaInversion::diff(COS_PSI0 + 0.3, params());
            let d_below = UmbrellaInversion::diff(COS_PSI0 - 0.3, params());

            assert!(d_above > 0.0);
            assert!(d_below < 0.0);
        }
    }
}
