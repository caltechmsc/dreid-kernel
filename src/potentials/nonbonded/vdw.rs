use crate::math::Real;
use crate::traits::PairKernel;
use crate::types::EnergyDiff;

/// Lennard-Jones 12-6 potential for Van der Waals interactions.
///
/// # Physics
///
/// Models the short-range repulsion and long-range attraction between non-bonded atoms.
///
/// - **Formula**: $$ E = D_0 \left[ \left( \frac{R_0}{r} \right)^{12} - 2 \left( \frac{R_0}{r} \right)^6 \right] $$
/// - **Derivative Factor (`diff`)**: $$ D = -\frac{1}{r} \frac{dE}{dr} = \frac{12 D_0}{r^2} \left[ \left( \frac{R_0}{r} \right)^{12} - \left( \frac{R_0}{r} \right)^6 \right] $$
///
/// # Parameters
///
/// - `d0`: The energy well depth $D_0$.
/// - `r0_sq`: The squared equilibrium distance $R_0^2$.
///
/// # Pre-computation
///
/// Use [`LennardJones::precompute`] to convert physical constants into optimized parameters:
/// $(D_0, R_0) \to (D_0, R_0^2)$.
///
/// # Inputs
///
/// - `r_sq`: Squared distance $r^2$ between two atoms.
///
/// # Implementation Notes
///
/// - This implementation avoids `sqrt` entirely by operating on squared distances.
/// - The power chain `s -> s3 -> s6` is used for efficient calculation of $r^{-6}$ and $r^{-12}$ terms.
/// - All intermediate calculations are shared between energy and force computations.
/// - Branchless and panic-free.
#[derive(Clone, Copy, Debug, Default)]
pub struct LennardJones;

impl LennardJones {
    /// Pre-computes optimized kernel parameters from physical constants.
    ///
    /// # Input
    ///
    /// - `d0`: Energy well depth $D_0$.
    /// - `r0`: Equilibrium distance $R_0$.
    ///
    /// # Output
    ///
    /// Returns `(d0, r0_sq)`:
    /// - `d0`: Well depth (passed through).
    /// - `r0_sq`: Squared equilibrium distance $R_0^2$.
    ///
    /// # Computation
    ///
    /// $$ R_0^2 = R_0 \times R_0 $$
    #[inline(always)]
    pub fn precompute<T: Real>(d0: T, r0: T) -> (T, T) {
        (d0, r0 * r0)
    }
}

impl<T: Real> PairKernel<T> for LennardJones {
    type Params = (T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = D_0 (s^6 - 2s^3), \quad \text{where } s = (R_0/r)^2 $$
    #[inline(always)]
    fn energy(r_sq: T, (d0, r0_sq): Self::Params) -> T {
        let s = r0_sq * r_sq.recip();
        let s3 = s * s * s;
        let s6 = s3 * s3;

        d0 * (s6 - T::from(2.0) * s3)
    }

    /// Computes only the force pre-factor $D$.
    ///
    /// # Formula
    ///
    /// $$ D = \frac{12 D_0}{r^2} (s^6 - s^3), \quad \text{where } s = (R_0/r)^2 $$
    ///
    /// This factor is defined such that the force vector can be computed
    /// by a single vector multiplication: $\vec{F} = -D \cdot \vec{r}$.
    #[inline(always)]
    fn diff(r_sq: T, (d0, r0_sq): Self::Params) -> T {
        let inv_r2 = r_sq.recip();
        let s = r0_sq * inv_r2;
        let s3 = s * s * s;
        let s6 = s3 * s3;

        T::from(12.0) * d0 * inv_r2 * (s6 - s3)
    }

    /// Computes both energy and force pre-factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(r_sq: T, (d0, r0_sq): Self::Params) -> EnergyDiff<T> {
        let inv_r2 = r_sq.recip();
        let s = r0_sq * inv_r2;
        let s3 = s * s * s;
        let s6 = s3 * s3;

        let energy = d0 * (s6 - T::from(2.0) * s3);

        let diff = T::from(12.0) * d0 * inv_r2 * (s6 - s3);

        EnergyDiff { energy, diff }
    }
}

/// Buckingham (Exponential-6) potential for Van der Waals interactions.
///
/// # Physics
///
/// Models short-range repulsion exponentially and long-range attraction with an $r^{-6}$ term,
/// providing a more physically accurate repulsion wall than Lennard-Jones.
///
/// - **Formula**: $$ E = D_0 \left[ \frac{6}{\zeta-6} \exp\left(\zeta(1 - \frac{r}{R_0})\right) - \frac{\zeta}{\zeta-6} \left(\frac{R_0}{r}\right)^6 \right] $$
/// - **Derivative Factor (`diff`)**: $$ D = -\frac{1}{r} \frac{dE}{dr} = \frac{6\zeta D_0}{r(\zeta-6)R_0} \left[ \exp\left(\zeta(1 - \frac{r}{R_0})\right) - \left(\frac{R_0}{r}\right)^7 \right] $$
///
/// # Parameters
///
/// For computational efficiency, the physical parameters ($D_0, R_0, \zeta$) are pre-computed
/// into the standard Buckingham form ($A, B, C$):
/// - `a`: The repulsion pre-factor $A = \frac{6 D_0}{\zeta-6} e^{\zeta}$.
/// - `b`: The repulsion decay constant $B = \zeta / R_0$.
/// - `c`: The attraction pre-factor $C = \frac{\zeta D_0 R_0^6}{\zeta-6}$.
/// - `r_max_sq`: The squared distance of the local energy maximum $r_{\text{max}}^2$.
/// - `two_e_max`: Twice the energy at the local maximum, $2 E(r_{\text{max}})$.
///
/// # Pre-computation
///
/// Use [`Buckingham::precompute`] to convert physical constants into optimized parameters:
/// $(D_0, R_0, \zeta) \to (A, B, C, r_{max}^2, 2E_{max})$.
/// This involves computing the $A, B, C$ form and finding the reflection point
/// via Newton's method.
///
/// # Inputs
///
/// - `r_sq`: Squared distance $r^2$ between two atoms.
///
/// # Implementation Notes
///
/// - The kernel operates on the computationally efficient $A, B, C$ form.
/// - For $r < r_{\text{max}}$, the energy is reflected about the local maximum:
///   $E_{\text{ref}}(r) = 2 E_{\text{max}} - E(r)$. This produces a repulsive wall
///   that diverges to $+\infty$ as $r \to 0$ via the $C/r^6$ attraction term,
///   while maintaining $C^1$ continuity at $r_{\text{max}}$ (where $E'(r_{\text{max}}) = 0$).
/// - A branchless sign-flip replaces the traditional constant penalty, providing
///   physically correct short-range behavior at zero additional runtime cost.
/// - Requires one `sqrt` and one `exp` call, making it computationally more demanding than LJ.
/// - Power chain `inv_r2 -> inv_r6 -> inv_r8` is used for the attractive term.
#[derive(Clone, Copy, Debug, Default)]
pub struct Buckingham;

impl Buckingham {
    /// Pre-computes optimized kernel parameters from physical constants.
    ///
    /// # Input
    ///
    /// - `d0`: Energy well depth $D_0$.
    /// - `r0`: Equilibrium distance $R_0$.
    /// - `zeta`: Steepness parameter $\zeta$ (must be $> 6$).
    ///
    /// # Output
    ///
    /// Returns `(a, b, c, r_max_sq, two_e_max)`:
    /// - `a`: Repulsion pre-factor $A = \frac{6 D_0}{\zeta-6} e^{\zeta}$.
    /// - `b`: Repulsion decay constant $B = \zeta / R_0$.
    /// - `c`: Attraction pre-factor $C = \frac{\zeta D_0 R_0^6}{\zeta-6}$.
    /// - `r_max_sq`: Squared distance of the local energy maximum.
    /// - `two_e_max`: Twice the energy at the local maximum.
    ///
    /// # Computation
    ///
    /// $$ A = \frac{6 D_0}{\zeta - 6} e^{\zeta}, \quad B = \frac{\zeta}{R_0}, \quad C = \frac{\zeta D_0 R_0^6}{\zeta - 6} $$
    ///
    /// The reflection point $r_{max}$ is found by solving $dE/dr = 0$ via Newton's method.
    #[inline(always)]
    pub fn precompute<T: Real>(d0: T, r0: T, zeta: T) -> (T, T, T, T, T) {
        let six = T::from(6.0);
        let zeta_minus_6 = zeta - six;

        let r0_2 = r0 * r0;
        let r0_3 = r0_2 * r0;
        let r0_6 = r0_3 * r0_3;

        let a = six * d0 * T::exp(zeta) / zeta_minus_6;
        let b = zeta / r0;
        let c = zeta * d0 * r0_6 / zeta_minus_6;

        let seven = T::from(7.0);
        let mut r = r0;
        for _ in 0..32 {
            let exp_term = T::exp(-b * r);
            let r2 = r * r;
            let r3 = r2 * r;
            let r6 = r3 * r3;
            let r7 = r6 * r;

            let g = a * b * exp_term * r7 - six * c;
            let gp = a * b * exp_term * r6 * (seven - b * r);
            r = r - g / gp;
        }

        let r_max_sq = r * r;

        let inv_r = r.recip();
        let inv_r2 = inv_r * inv_r;
        let inv_r3 = inv_r2 * inv_r;
        let inv_r6 = inv_r3 * inv_r3;
        let e_max = a * T::exp(-b * r) - c * inv_r6;
        let two_e_max = e_max + e_max;

        (a, b, c, r_max_sq, two_e_max)
    }
}

impl<T: Real> PairKernel<T> for Buckingham {
    type Params = (T, T, T, T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E(r) = \begin{cases} A e^{-Br} - C/r^6 & r \ge r_{\text{max}} \\ 2 E_{\text{max}} - (A e^{-Br} - C/r^6) & r < r_{\text{max}} \end{cases} $$
    #[inline(always)]
    fn energy(r_sq: T, (a, b, c, r_max_sq, two_e_max): Self::Params) -> T {
        let mask = T::from((r_sq >= r_max_sq) as u8 as f32);
        let sign = mask + mask - T::from(1.0);

        let r = r_sq.sqrt();
        let inv_r2 = r_sq.recip();
        let inv_r6 = inv_r2 * inv_r2 * inv_r2;

        let e_buck = a * T::exp(-b * r) - c * inv_r6;

        sign * e_buck + (T::from(1.0) - mask) * two_e_max
    }

    /// Computes only the force pre-factor $D$.
    ///
    /// # Formula
    ///
    /// $$ D(r) = \text{sign}(r) \left( \frac{A B e^{-Br}}{r} - \frac{6C}{r^8} \right) $$
    ///
    /// where $\text{sign}(r) = +1$ for $r \ge r_{\text{max}}$ and $-1$ otherwise.
    /// At the maximum, $D(r_{\text{max}}) = 0$ from both sides, ensuring $C^1$ continuity.
    ///
    /// This factor is defined such that the force vector can be computed
    /// by a single vector multiplication: $\vec{F} = -D \cdot \vec{r}$.
    #[inline(always)]
    fn diff(r_sq: T, (a, b, c, r_max_sq, _two_e_max): Self::Params) -> T {
        let mask = T::from((r_sq >= r_max_sq) as u8 as f32);
        let sign = mask + mask - T::from(1.0);

        let inv_r = r_sq.rsqrt();
        let r = r_sq * inv_r;
        let inv_r2 = inv_r * inv_r;
        let inv_r4 = inv_r2 * inv_r2;
        let inv_r8 = inv_r4 * inv_r4;

        let exp_term = T::exp(-b * r);
        let d_buck = a * b * exp_term * inv_r - T::from(6.0) * c * inv_r8;

        sign * d_buck
    }

    /// Computes both energy and force pre-factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(r_sq: T, (a, b, c, r_max_sq, two_e_max): Self::Params) -> EnergyDiff<T> {
        let mask = T::from((r_sq >= r_max_sq) as u8 as f32);
        let sign = mask + mask - T::from(1.0);

        let inv_r = r_sq.rsqrt();
        let r = r_sq * inv_r;
        let inv_r2 = inv_r * inv_r;
        let inv_r4 = inv_r2 * inv_r2;
        let inv_r6 = inv_r4 * inv_r2;
        let inv_r8 = inv_r6 * inv_r2;

        let exp_term = T::exp(-b * r);

        let e_buck = a * exp_term - c * inv_r6;
        let d_buck = a * b * exp_term * inv_r - T::from(6.0) * c * inv_r8;

        EnergyDiff {
            energy: sign * e_buck + (T::from(1.0) - mask) * two_e_max,
            diff: sign * d_buck,
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

    // Typical LJ parameters: D0 = 0.1 kcal/mol, R0 = 3.5 Å
    const D0: f64 = 0.1;
    const R0: f64 = 3.5;
    const R0_SQ: f64 = R0 * R0;

    // Buckingham parameters: A, B, C
    const BUCK_A: f64 = 1000.0;
    const BUCK_B: f64 = 3.0;
    const BUCK_C: f64 = 50.0;

    // ========================================================================
    // Lennard-Jones Tests
    // ========================================================================

    mod lennard_jones {
        use super::*;

        const D0: f64 = 0.1;
        const R0: f64 = 3.5;
        const R0_SQ: f64 = R0 * R0;

        fn params() -> (f64, f64) {
            LennardJones::precompute(D0, R0)
        }

        // --------------------------------------------------------------------
        // 1. Sanity Checks
        // --------------------------------------------------------------------

        #[test]
        fn sanity_compute_equals_separate() {
            let r_sq = 9.0_f64;
            let p = params();

            let result = LennardJones::compute(r_sq, p);
            let energy_only = LennardJones::energy(r_sq, p);
            let diff_only = LennardJones::diff(r_sq, p);

            assert_relative_eq!(result.energy, energy_only, epsilon = 1e-14);
            assert_relative_eq!(result.diff, diff_only, epsilon = 1e-14);
        }

        #[test]
        fn sanity_f32_f64_consistency() {
            let r_sq = 12.25;
            let p64 = params();
            let p32 = LennardJones::precompute(D0 as f32, R0 as f32);

            let e64 = LennardJones::energy(r_sq, p64);
            let e32 = LennardJones::energy(r_sq as f32, p32);

            assert_relative_eq!(e64, e32 as f64, epsilon = 1e-5);
        }

        #[test]
        fn sanity_equilibrium_energy_minimum() {
            let e = LennardJones::energy(R0_SQ, params());
            assert_relative_eq!(e, -D0, epsilon = 1e-10);
        }

        #[test]
        fn sanity_equilibrium_zero_force() {
            let d = LennardJones::diff(R0_SQ, params());
            assert_relative_eq!(d, 0.0, epsilon = 1e-10);
        }

        // --------------------------------------------------------------------
        // 2. Numerical Stability
        // --------------------------------------------------------------------

        #[test]
        fn stability_large_distance() {
            let r_sq = 1e6_f64;
            let result = LennardJones::compute(r_sq, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
            assert!(result.energy.abs() < 1e-10);
        }

        #[test]
        fn stability_small_distance() {
            let r_sq = 1.0_f64;
            let result = LennardJones::compute(r_sq, params());

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
            let e_plus = LennardJones::energy(r_plus * r_plus, p);
            let e_minus = LennardJones::energy(r_minus * r_minus, p);
            let de_dr_numerical = (e_plus - e_minus) / (2.0 * H);

            let d_analytic = LennardJones::diff(r_sq, p);
            let de_dr_analytic = -d_analytic * r;

            assert_relative_eq!(de_dr_numerical, de_dr_analytic, epsilon = TOL_DIFF);
        }

        #[test]
        fn finite_diff_repulsion_region() {
            finite_diff_check(2.5);
        }

        #[test]
        fn finite_diff_equilibrium_region() {
            finite_diff_check(R0);
        }

        #[test]
        fn finite_diff_attraction_region() {
            finite_diff_check(5.0);
        }

        #[test]
        fn finite_diff_long_range() {
            finite_diff_check(10.0);
        }

        // --------------------------------------------------------------------
        // 4. LJ-Specific: Physical Behavior
        // --------------------------------------------------------------------

        #[test]
        fn specific_repulsion_positive_energy() {
            let e = LennardJones::energy(4.0, params());
            assert!(e > 0.0);
        }

        #[test]
        fn specific_attraction_negative_energy() {
            let e = LennardJones::energy(25.0, params());
            assert!(e < 0.0);
        }

        #[test]
        fn specific_diff_sign_repulsion() {
            let d = LennardJones::diff(4.0, params());
            assert!(d > 0.0);
        }

        #[test]
        fn specific_diff_sign_attraction() {
            let d = LennardJones::diff(25.0, params());
            assert!(d < 0.0);
        }

        // --------------------------------------------------------------------
        // 5. Precompute
        // --------------------------------------------------------------------

        #[test]
        fn precompute_values() {
            let (d0, r0_sq) = LennardJones::precompute(D0, R0);
            assert_relative_eq!(d0, D0, epsilon = 1e-14);
            assert_relative_eq!(r0_sq, R0_SQ, epsilon = 1e-14);
        }

        #[test]
        fn precompute_round_trip() {
            let p = LennardJones::precompute(D0, R0);
            let e = LennardJones::energy(R0_SQ, p);
            assert_relative_eq!(e, -D0, epsilon = 1e-10);
        }
    }

    // ========================================================================
    // Buckingham Tests
    // ========================================================================

    mod buckingham {
        use super::*;

        /// Computes the local maximum of the Buckingham potential via Newton's method.
        fn reflection_params(a: f64, b: f64, c: f64) -> (f64, f64) {
            let mut r = 1.0_f64;
            for _ in 0..100 {
                let exp_term = (-b * r).exp();
                let r7 = r.powi(7);
                let g = a * b * exp_term * r7 - 6.0 * c;
                let gp = a * b * exp_term * r.powi(6) * (7.0 - b * r);
                r -= g / gp;
            }
            let e_max = a * (-b * r).exp() - c / r.powi(6);
            (r * r, 2.0 * e_max)
        }

        fn params() -> (f64, f64, f64, f64, f64) {
            let (r_max_sq, two_e_max) = reflection_params(BUCK_A, BUCK_B, BUCK_C);
            (BUCK_A, BUCK_B, BUCK_C, r_max_sq, two_e_max)
        }

        // --------------------------------------------------------------------
        // 1. Sanity Checks
        // --------------------------------------------------------------------

        #[test]
        fn sanity_compute_equals_separate() {
            let r_sq = 4.0_f64;
            let p = params();

            let result = Buckingham::compute(r_sq, p);
            let energy_only = Buckingham::energy(r_sq, p);
            let diff_only = Buckingham::diff(r_sq, p);

            assert_relative_eq!(result.energy, energy_only, epsilon = 1e-12);
            assert_relative_eq!(result.diff, diff_only, epsilon = 1e-12);
        }

        #[test]
        fn sanity_compute_equals_separate_reflected() {
            let p = params();
            let r_sq = 0.25_f64;

            let result = Buckingham::compute(r_sq, p);
            let energy_only = Buckingham::energy(r_sq, p);
            let diff_only = Buckingham::diff(r_sq, p);

            assert_relative_eq!(result.energy, energy_only, epsilon = 1e-12);
            assert_relative_eq!(result.diff, diff_only, epsilon = 1e-12);
        }

        #[test]
        fn sanity_f32_f64_consistency() {
            let r_sq = 4.0;
            let p64 = params();
            let (r_max_sq_32, two_e_max_32) = reflection_params(BUCK_A, BUCK_B, BUCK_C);
            let p32 = (
                BUCK_A as f32,
                BUCK_B as f32,
                BUCK_C as f32,
                r_max_sq_32 as f32,
                two_e_max_32 as f32,
            );

            let e64 = Buckingham::energy(r_sq, p64);
            let e32 = Buckingham::energy(r_sq as f32, p32);

            assert_relative_eq!(e64, e32 as f64, epsilon = 1e-3);
        }

        // --------------------------------------------------------------------
        // 2. Numerical Stability
        // --------------------------------------------------------------------

        #[test]
        fn stability_reflected_region() {
            let r_sq = 0.1_f64;
            let result = Buckingham::compute(r_sq, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
            assert!(result.energy > 0.0);
            assert!(result.diff > 0.0);
        }

        #[test]
        fn stability_large_distance() {
            let r_sq = 1e4_f64;
            let result = Buckingham::compute(r_sq, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
        }

        #[test]
        fn stability_near_zero() {
            let r_sq = 1e-20_f64;
            let p = params();
            let e = Buckingham::energy(r_sq, p);

            assert!(e.is_finite());
            assert!(e > 0.0);
        }

        // --------------------------------------------------------------------
        // 3. Finite Difference Verification
        // --------------------------------------------------------------------

        fn finite_diff_check(r: f64) {
            let p = params();
            let r_sq = r * r;

            let r_plus = r + H;
            let r_minus = r - H;
            let e_plus = Buckingham::energy(r_plus * r_plus, p);
            let e_minus = Buckingham::energy(r_minus * r_minus, p);
            let de_dr_numerical = (e_plus - e_minus) / (2.0 * H);

            let d_analytic = Buckingham::diff(r_sq, p);
            let de_dr_analytic = -d_analytic * r;

            assert_relative_eq!(de_dr_numerical, de_dr_analytic, epsilon = TOL_DIFF);
        }

        #[test]
        fn finite_diff_reflected_region() {
            finite_diff_check(0.8);
        }

        #[test]
        fn finite_diff_normal_short_range() {
            finite_diff_check(1.5);
        }

        #[test]
        fn finite_diff_medium_range() {
            finite_diff_check(3.0);
        }

        #[test]
        fn finite_diff_long_range() {
            finite_diff_check(8.0);
        }

        // --------------------------------------------------------------------
        // 4. Buckingham-Specific: Reflection Properties
        // --------------------------------------------------------------------

        #[test]
        fn specific_reflection_diverges() {
            let p = params();
            let e_close = Buckingham::energy(0.01, p);
            let e_far = Buckingham::energy(0.25, p);
            assert!(e_close > e_far);
        }

        #[test]
        fn specific_diff_at_maximum_is_zero() {
            let p = params();
            let r_max_sq = p.3;
            let d = Buckingham::diff(r_max_sq, p);
            assert_relative_eq!(d, 0.0, epsilon = 1e-6);
        }

        #[test]
        fn specific_c1_continuity_at_maximum() {
            let p = params();
            let r_max = p.3.sqrt();
            let eps = 1e-8;

            let r_inside = r_max - eps;
            let r_outside = r_max + eps;

            let d_inside = Buckingham::diff(r_inside * r_inside, p);
            let d_outside = Buckingham::diff(r_outside * r_outside, p);

            let de_dr_inside = -d_inside * r_inside;
            let de_dr_outside = -d_outside * r_outside;

            assert_relative_eq!(de_dr_inside, de_dr_outside, epsilon = 1e-3);
        }

        #[test]
        fn specific_energy_continuity_at_maximum() {
            let p = params();
            let r_max = p.3.sqrt();
            let eps = 1e-8;

            let e_inside = Buckingham::energy((r_max - eps).powi(2), p);
            let e_outside = Buckingham::energy((r_max + eps).powi(2), p);

            assert_relative_eq!(e_inside, e_outside, epsilon = 1e-4);
        }

        #[test]
        fn specific_finite_diff_across_boundary() {
            let p = params();
            let r_max = p.3.sqrt();

            let h = 1e-6;

            let r_out = r_max + 0.01;
            let e_p = Buckingham::energy((r_out + h).powi(2), p);
            let e_m = Buckingham::energy((r_out - h).powi(2), p);
            let de_dr_num_out = (e_p - e_m) / (2.0 * h);
            let de_dr_ana_out = -Buckingham::diff(r_out * r_out, p) * r_out;
            assert_relative_eq!(de_dr_num_out, de_dr_ana_out, epsilon = TOL_DIFF);

            let r_in = r_max - 0.01;
            let e_p = Buckingham::energy((r_in + h).powi(2), p);
            let e_m = Buckingham::energy((r_in - h).powi(2), p);
            let de_dr_num_in = (e_p - e_m) / (2.0 * h);
            let de_dr_ana_in = -Buckingham::diff(r_in * r_in, p) * r_in;
            assert_relative_eq!(de_dr_num_in, de_dr_ana_in, epsilon = TOL_DIFF);
        }

        // --------------------------------------------------------------------
        // 5. Precompute
        // --------------------------------------------------------------------

        #[test]
        fn precompute_values() {
            let (a, b, c, r_max_sq, two_e_max) = Buckingham::precompute(1.0, 2.0, 12.0);
            assert_relative_eq!(a, 12.0_f64.exp(), epsilon = 1e-4);
            assert_relative_eq!(b, 6.0, epsilon = 1e-14);
            assert_relative_eq!(c, 128.0, epsilon = 1e-10);
            assert!(r_max_sq > 0.0);
            assert!(two_e_max.is_finite());
        }

        #[test]
        fn precompute_round_trip() {
            let p = Buckingham::precompute(1.0, 2.0, 12.0);
            let e = Buckingham::energy(4.0, p);
            assert_relative_eq!(e, -1.0, epsilon = 1e-6);
        }
    }
}
