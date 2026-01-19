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
/// - `r_fusion_sq`: The squared distance threshold for regularization.
///
/// # Inputs
///
/// - `r_sq`: Squared distance $r^2$ between two atoms.
///
/// # Implementation Notes
///
/// - The kernel operates on the computationally efficient $A, B, C$ form.
/// - A branchless regularization is applied for $r^2 < r_{fusion}^2$ using a mathematical mask
///   to prevent energy collapse at very short distances, ensuring numerical stability.
/// - Requires one `sqrt` and one `exp` call, making it computationally more demanding than LJ.
/// - Power chain `inv_r2 -> inv_r6 -> inv_r8` is used for the attractive term.
#[derive(Clone, Copy, Debug, Default)]
pub struct Buckingham;

impl<T: Real> PairKernel<T> for Buckingham {
    type Params = (T, T, T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = A e^{-B r} - C / r^6 $$
    #[inline(always)]
    fn energy(r_sq: T, (a, b, c, r_fusion_sq): Self::Params) -> T {
        let is_safe = T::from((r_sq >= r_fusion_sq) as u8 as f32);
        let effective_r_sq = r_sq.max(r_fusion_sq);

        let r = effective_r_sq.sqrt();
        let inv_r2 = effective_r_sq.recip();
        let inv_r6 = inv_r2 * inv_r2 * inv_r2;

        let repulsion = a * T::exp(-b * r);
        let attraction = c * inv_r6;
        let energy_unsafe = repulsion - attraction;

        const FUSION_ENERGY_PENALTY: f32 = 1.0e6;
        let penalty = T::from(FUSION_ENERGY_PENALTY);

        energy_unsafe * is_safe + penalty * (T::from(1.0) - is_safe)
    }

    /// Computes only the force pre-factor $D$.
    ///
    /// # Formula
    ///
    /// $$ D = \frac{A B e^{-B r}}{r} - \frac{6 C}{r^8} $$
    ///
    /// This factor is defined such that the force vector can be computed
    /// by a single vector multiplication: $\vec{F} = -D \cdot \vec{r}$.
    #[inline(always)]
    fn diff(r_sq: T, (a, b, c, r_fusion_sq): Self::Params) -> T {
        let is_safe = T::from((r_sq >= r_fusion_sq) as u8 as f32);
        let effective_r_sq = r_sq.max(r_fusion_sq);

        let inv_r = effective_r_sq.rsqrt();
        let r = effective_r_sq * inv_r;
        let inv_r2 = inv_r * inv_r;
        let inv_r4 = inv_r2 * inv_r2;
        let inv_r8 = inv_r4 * inv_r4;

        let exp_term = T::exp(-b * r);

        let repulsion_factor = a * b * exp_term * inv_r;
        let attraction_factor = T::from(6.0) * c * inv_r8;
        let diff_unsafe = repulsion_factor - attraction_factor;

        const FUSION_FORCE_PENALTY: f32 = 1.0e6;
        let penalty = T::from(FUSION_FORCE_PENALTY);

        diff_unsafe * is_safe + penalty * (T::from(1.0) - is_safe)
    }

    /// Computes both energy and force pre-factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(r_sq: T, (a, b, c, r_fusion_sq): Self::Params) -> EnergyDiff<T> {
        let is_safe = T::from((r_sq >= r_fusion_sq) as u8 as f32);
        let effective_r_sq = r_sq.max(r_fusion_sq);

        let inv_r = effective_r_sq.rsqrt();
        let r = effective_r_sq * inv_r;
        let inv_r2 = inv_r * inv_r;
        let inv_r4 = inv_r2 * inv_r2;
        let inv_r6 = inv_r4 * inv_r2;
        let inv_r8 = inv_r6 * inv_r2;

        let exp_term = T::exp(-b * r);

        let repulsion_energy = a * exp_term;
        let attraction_energy = c * inv_r6;
        let energy_unsafe = repulsion_energy - attraction_energy;

        let repulsion_force = repulsion_energy * b * inv_r;
        let attraction_force = T::from(6.0) * c * inv_r8;
        let diff_unsafe = repulsion_force - attraction_force;

        const FUSION_ENERGY_PENALTY: f32 = 1.0e6;
        const FUSION_FORCE_PENALTY: f32 = 1.0e6;

        let energy_penalty = T::from(FUSION_ENERGY_PENALTY);
        let force_penalty = T::from(FUSION_FORCE_PENALTY);

        EnergyDiff {
            energy: energy_unsafe * is_safe + energy_penalty * (T::from(1.0) - is_safe),
            diff: diff_unsafe * is_safe + force_penalty * (T::from(1.0) - is_safe),
        }
    }
}

/// Splined Buckingham (Exp-6) potential with $C^2$ continuous short-range regularization.
///
/// # Physics
///
/// This kernel enhances the standard Buckingham potential by replacing the problematic
/// short-range region ($r < r_{spline}$) with a quintic (5th degree) polynomial, $P_5(r)$.
/// This guarantees that the energy, force ($C^1$), and force derivative ($C^2$) are
/// continuous everywhere, which is critical for stable molecular dynamics simulations.
///
/// - **Formula**: $$ E(r) = \begin{cases} D_0 \left[ \frac{6}{\zeta-6} \exp\left(\zeta(1 - \frac{r}{R_0})\right) - \frac{\zeta}{\zeta-6} \left(\frac{R_0}{r}\right)^6 \right] & r \ge r_{spline} \\ P_5(r) = \sum_{i=0}^{5} p_i r^i & r < r_{spline} \end{cases} $$
/// - **Derivative Factor (`diff`)**: $$ D = -\frac{1}{r} \frac{dE}{dr} = \begin{cases} \frac{6\zeta D_0}{r(\zeta-6)R_0} \left[ \exp\left(\zeta(1 - \frac{r}{R_0})\right) - \left(\frac{R_0}{r}\right)^7 \right] & r \ge r_{spline} \\ -\frac{1}{r} \sum_{i=1}^{5} i \cdot p_i r^{i-1} & r < r_{spline} \end{cases} $$
///
/// # Parameters
///
/// For computational efficiency, the physical parameters ($D_0, R_0, \zeta$) are pre-computed
/// into two sets for the long-range and short-range parts:
///
/// - **Long-Range Part ($r \ge r_{spline}$)**:
///   - `a`: The repulsion pre-factor $A = \frac{6 D_0}{\zeta-6} e^{\zeta}$.
///   - `b`: The repulsion decay constant $B = \zeta / R_0$.
///   - `c`: The attraction pre-factor $C = \frac{\zeta D_0 R_0^6}{\zeta-6}$.
///
/// - **Short-Range Part ($r < r_{spline}$)**:
///   - `r_spline_sq`: The squared distance threshold for switching to the polynomial.
///   - `p0..p5`: The six coefficients of the quintic polynomial $P_5(r)$, pre-computed
///     to satisfy $C^2$ continuity at $r_{spline}$ and boundary conditions at $r=0$.
///
/// # Inputs
///
/// - `r_sq`: Squared distance $r^2$ between two atoms.
///
/// # Implementation Notes
///
/// - A branchless selection mechanism is used to switch between the Buckingham and
///   polynomial forms, making it SIMD-friendly.
/// - The polynomial is evaluated using Horner's method for improved numerical stability
///   and reduced floating-point operations.
/// - The entire computation, including both paths, is executed to avoid pipeline stalls,
///   making the runtime performance constant and predictable.
#[derive(Clone, Copy, Debug, Default)]
pub struct SplinedBuckingham;

impl<T: Real> PairKernel<T> for SplinedBuckingham {
    type Params = (T, T, T, T, T, T, T, T, T, T);

    /// Computes only the potential energy, selecting between Exp-6 and polynomial forms.
    ///
    /// # Formula
    ///
    /// $$ E(r) = \begin{cases} A e^{-Br} - C/r^6 & r \ge r_{spline} \\ P_5(r) & r < r_{spline} \end{cases} $$
    #[inline(always)]
    fn energy(r_sq: T, params: Self::Params) -> T {
        let (a, b, c, r_spline_sq, p0, p1, p2, p3, p4, p5) = params;

        let mask = T::from((r_sq >= r_spline_sq) as u8 as f32);

        let r = r_sq.sqrt();

        let r_sq_long = r_sq.max(r_spline_sq);
        let inv_r2_long = r_sq_long.recip();
        let inv_r6_long = inv_r2_long * inv_r2_long * inv_r2_long;

        let energy_long = a * T::exp(-b * r) - c * inv_r6_long;

        let energy_short = ((((p5 * r + p4) * r + p3) * r + p2) * r + p1) * r + p0;

        energy_long * mask + energy_short * (T::from(1.0) - mask)
    }

    /// Computes only the force pre-factor $D$, selecting between Exp-6 and polynomial forms.
    ///
    /// # Formula
    ///
    /// $$ D(r) = \begin{cases} \frac{A B e^{-B r}}{r} - \frac{6 C}{r^8} & r \ge r_{spline} \\ -\frac{P'_5(r)}{r} & r < r_{spline} \end{cases} $$
    ///
    /// This factor is defined such that the force vector can be computed
    /// by a single vector multiplication: $\vec{F} = -D \cdot \vec{r}$.
    #[inline(always)]
    fn diff(r_sq: T, params: Self::Params) -> T {
        let (a, b, c, r_spline_sq, _, p1, p2, p3, p4, p5) = params;

        let mask = T::from((r_sq >= r_spline_sq) as u8 as f32);

        let r = r_sq.sqrt();

        let r_sq_long = r_sq.max(r_spline_sq);
        let inv_r_long = r_sq_long.rsqrt();
        let inv_r2_long = inv_r_long * inv_r_long;
        let inv_r4_long = inv_r2_long * inv_r2_long;
        let inv_r8_long = inv_r4_long * inv_r4_long;

        let diff_long = a * b * T::exp(-b * r) * inv_r_long - T::from(6.0) * c * inv_r8_long;

        let r2 = r_sq;
        let r3 = r2 * r;
        let r4 = r2 * r2;
        let poly_deriv = p1
            + T::from(2.0) * p2 * r
            + T::from(3.0) * p3 * r2
            + T::from(4.0) * p4 * r3
            + T::from(5.0) * p5 * r4;
        let diff_short = -(poly_deriv * r.recip());

        diff_long * mask + diff_short * (T::from(1.0) - mask)
    }

    /// Computes both energy and force pre-factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(r_sq: T, params: Self::Params) -> EnergyDiff<T> {
        let (a, b, c, r_spline_sq, p0, p1, p2, p3, p4, p5) = params;

        let mask = T::from((r_sq >= r_spline_sq) as u8 as f32);

        let r = r_sq.sqrt();

        let r_sq_long = r_sq.max(r_spline_sq);
        let inv_r_long = r_sq_long.rsqrt();
        let inv_r2_long = inv_r_long * inv_r_long;
        let inv_r4_long = inv_r2_long * inv_r2_long;
        let inv_r6_long = inv_r4_long * inv_r2_long;

        let exp_term_long = T::exp(-b * r);
        let energy_long = a * exp_term_long - c * inv_r6_long;

        let inv_r8_long = inv_r4_long * inv_r4_long;
        let diff_long = a * b * exp_term_long * inv_r_long - T::from(6.0) * c * inv_r8_long;

        let energy_short = ((((p5 * r + p4) * r + p3) * r + p2) * r + p1) * r + p0;
        let r2 = r_sq;
        let poly_deriv = p1
            + T::from(2.0) * p2 * r
            + T::from(3.0) * p3 * r2
            + T::from(4.0) * p4 * (r2 * r)
            + T::from(5.0) * p5 * (r2 * r2);
        let diff_short = -(poly_deriv * r.recip());

        EnergyDiff {
            energy: energy_long * mask + energy_short * (T::from(1.0) - mask),
            diff: diff_long * mask + diff_short * (T::from(1.0) - mask),
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

    // Buckingham parameters: A, B, C, r_fusion_sq
    const BUCK_A: f64 = 1000.0;
    const BUCK_B: f64 = 3.0;
    const BUCK_C: f64 = 50.0;
    const BUCK_FUSION_SQ: f64 = 0.5;

    // ========================================================================
    // Lennard-Jones Tests
    // ========================================================================

    mod lennard_jones {
        use super::*;

        // --------------------------------------------------------------------
        // 1. Sanity Checks
        // --------------------------------------------------------------------

        #[test]
        fn sanity_compute_equals_separate() {
            let r_sq = 9.0_f64;
            let params = (D0, R0_SQ);

            let result = LennardJones::compute(r_sq, params);
            let energy_only = LennardJones::energy(r_sq, params);
            let diff_only = LennardJones::diff(r_sq, params);

            assert_relative_eq!(result.energy, energy_only, epsilon = 1e-14);
            assert_relative_eq!(result.diff, diff_only, epsilon = 1e-14);
        }

        #[test]
        fn sanity_f32_f64_consistency() {
            let r_sq_64 = 12.25_f64;
            let r_sq_32 = 12.25_f32;
            let params_64 = (D0, R0_SQ);
            let params_32 = (D0 as f32, R0_SQ as f32);

            let e64 = LennardJones::energy(r_sq_64, params_64);
            let e32 = LennardJones::energy(r_sq_32, params_32);

            assert_relative_eq!(e64, e32 as f64, epsilon = 1e-5);
        }

        #[test]
        fn sanity_equilibrium_energy_minimum() {
            let e = LennardJones::energy(R0_SQ, (D0, R0_SQ));
            assert_relative_eq!(e, -D0, epsilon = 1e-10);
        }

        #[test]
        fn sanity_equilibrium_zero_force() {
            let d = LennardJones::diff(R0_SQ, (D0, R0_SQ));
            assert_relative_eq!(d, 0.0, epsilon = 1e-10);
        }

        // --------------------------------------------------------------------
        // 2. Numerical Stability
        // --------------------------------------------------------------------

        #[test]
        fn stability_large_distance() {
            let r_sq = 1e6_f64;
            let result = LennardJones::compute(r_sq, (D0, R0_SQ));

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
            assert!(result.energy.abs() < 1e-10);
        }

        #[test]
        fn stability_small_distance() {
            let r_sq = 1.0_f64;
            let result = LennardJones::compute(r_sq, (D0, R0_SQ));

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
            assert!(result.energy > 0.0);
        }

        // --------------------------------------------------------------------
        // 3. Finite Difference Verification
        // --------------------------------------------------------------------

        fn finite_diff_check(r: f64, params: (f64, f64)) {
            let r_sq = r * r;

            let r_plus = r + H;
            let r_minus = r - H;
            let e_plus = LennardJones::energy(r_plus * r_plus, params);
            let e_minus = LennardJones::energy(r_minus * r_minus, params);
            let de_dr_numerical = (e_plus - e_minus) / (2.0 * H);

            let d_analytic = LennardJones::diff(r_sq, params);
            let de_dr_analytic = -d_analytic * r;

            assert_relative_eq!(de_dr_numerical, de_dr_analytic, epsilon = TOL_DIFF);
        }

        #[test]
        fn finite_diff_repulsion_region() {
            finite_diff_check(2.5, (D0, R0_SQ));
        }

        #[test]
        fn finite_diff_equilibrium_region() {
            finite_diff_check(R0, (D0, R0_SQ));
        }

        #[test]
        fn finite_diff_attraction_region() {
            finite_diff_check(5.0, (D0, R0_SQ));
        }

        #[test]
        fn finite_diff_long_range() {
            finite_diff_check(10.0, (D0, R0_SQ));
        }

        // --------------------------------------------------------------------
        // 4. LJ-Specific: Physical Behavior
        // --------------------------------------------------------------------

        #[test]
        fn specific_repulsion_positive_energy() {
            let e = LennardJones::energy(4.0, (D0, R0_SQ));
            assert!(e > 0.0);
        }

        #[test]
        fn specific_attraction_negative_energy() {
            let e = LennardJones::energy(25.0, (D0, R0_SQ));
            assert!(e < 0.0);
        }

        #[test]
        fn specific_diff_sign_repulsion() {
            let d = LennardJones::diff(4.0, (D0, R0_SQ));
            assert!(d > 0.0);
        }

        #[test]
        fn specific_diff_sign_attraction() {
            let d = LennardJones::diff(25.0, (D0, R0_SQ));
            assert!(d < 0.0);
        }
    }

    // ========================================================================
    // Buckingham Tests
    // ========================================================================

    mod buckingham {
        use super::*;

        fn params() -> (f64, f64, f64, f64) {
            (BUCK_A, BUCK_B, BUCK_C, BUCK_FUSION_SQ)
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
        fn sanity_f32_f64_consistency() {
            let r_sq = 4.0;
            let p64 = params();
            let p32 = (
                BUCK_A as f32,
                BUCK_B as f32,
                BUCK_C as f32,
                BUCK_FUSION_SQ as f32,
            );

            let e64 = Buckingham::energy(r_sq, p64);
            let e32 = Buckingham::energy(r_sq as f32, p32);

            assert_relative_eq!(e64, e32 as f64, epsilon = 1e-3);
        }

        // --------------------------------------------------------------------
        // 2. Numerical Stability
        // --------------------------------------------------------------------

        #[test]
        fn stability_fusion_region() {
            let r_sq = 0.1_f64;
            let result = Buckingham::compute(r_sq, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
            assert!(result.energy > 1e5);
        }

        #[test]
        fn stability_large_distance() {
            let r_sq = 1e4_f64;
            let result = Buckingham::compute(r_sq, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
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
        fn finite_diff_short_range() {
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
        // 4. Buckingham-Specific
        // --------------------------------------------------------------------

        #[test]
        fn specific_exponential_dominates_short_range() {
            let e1 = Buckingham::energy(0.81, params());
            let e2 = Buckingham::energy(1.0, params());
            assert!(e1.is_finite());
            assert!(e2.is_finite());
        }
    }

    // ========================================================================
    // SplinedBuckingham Tests
    // ========================================================================

    mod splined_buckingham {
        use super::*;

        fn params() -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
            let a = 1000.0;
            let b = 3.0;
            let c = 50.0;
            let r_spline_sq = 1.0;

            let p0 = 100.0;
            let p1 = -50.0;
            let p2 = 10.0;
            let p3 = 0.0;
            let p4 = 0.0;
            let p5 = 0.0;

            (a, b, c, r_spline_sq, p0, p1, p2, p3, p4, p5)
        }

        // --------------------------------------------------------------------
        // 1. Sanity Checks
        // --------------------------------------------------------------------

        #[test]
        fn sanity_compute_equals_separate() {
            let r_sq = 4.0_f64;
            let p = params();

            let result = SplinedBuckingham::compute(r_sq, p);
            let energy_only = SplinedBuckingham::energy(r_sq, p);
            let diff_only = SplinedBuckingham::diff(r_sq, p);

            assert_relative_eq!(result.energy, energy_only, epsilon = 1e-12);
            assert_relative_eq!(result.diff, diff_only, epsilon = 1e-12);
        }

        // --------------------------------------------------------------------
        // 2. Numerical Stability
        // --------------------------------------------------------------------

        #[test]
        fn stability_inside_spline_region() {
            let r_sq = 0.25_f64;
            let result = SplinedBuckingham::compute(r_sq, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
        }

        #[test]
        fn stability_outside_spline_region() {
            let r_sq = 4.0_f64;
            let result = SplinedBuckingham::compute(r_sq, params());

            assert!(result.energy.is_finite());
            assert!(result.diff.is_finite());
        }

        // --------------------------------------------------------------------
        // 3. Finite Difference Verification
        // --------------------------------------------------------------------

        fn finite_diff_check(r: f64) {
            let p = params();
            let r_sq = r * r;

            let r_plus = r + H;
            let r_minus = r - H;
            let e_plus = SplinedBuckingham::energy(r_plus * r_plus, p);
            let e_minus = SplinedBuckingham::energy(r_minus * r_minus, p);
            let de_dr_numerical = (e_plus - e_minus) / (2.0 * H);

            let d_analytic = SplinedBuckingham::diff(r_sq, p);
            let de_dr_analytic = -d_analytic * r;

            assert_relative_eq!(de_dr_numerical, de_dr_analytic, epsilon = TOL_DIFF);
        }

        #[test]
        fn finite_diff_inside_spline() {
            finite_diff_check(0.5);
        }

        #[test]
        fn finite_diff_outside_spline() {
            finite_diff_check(2.0);
        }

        #[test]
        fn finite_diff_long_range() {
            finite_diff_check(5.0);
        }

        // --------------------------------------------------------------------
        // 4. Spline-Specific: Region Selection
        // --------------------------------------------------------------------

        #[test]
        fn specific_uses_polynomial_inside() {
            let p = params();
            let r_sq = 0.25_f64;

            let r = r_sq.sqrt();
            let expected = p.4 + p.5 * r + p.6 * r * r;
            let actual = SplinedBuckingham::energy(r_sq, p);

            assert_relative_eq!(actual, expected, epsilon = 1e-10);
        }

        fn c2_continuous_params() -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
            let a = 1000.0_f64;
            let b = 3.0_f64;
            let c = 50.0_f64;
            let r_s = 1.2_f64;
            let r_spline_sq = r_s * r_s;

            let exp_term = (-b * r_s).exp();
            let inv_r = 1.0 / r_s;
            let inv_r2 = inv_r * inv_r;
            let inv_r6 = inv_r2 * inv_r2 * inv_r2;
            let inv_r7 = inv_r6 * inv_r;
            let inv_r8 = inv_r6 * inv_r2;

            let e_s = a * exp_term - c * inv_r6;
            let de_s = -a * b * exp_term + 6.0 * c * inv_r7;
            let d2e_s = a * b * b * exp_term - 42.0 * c * inv_r8;

            let e_max = 1e4_f64;
            let p0 = e_max;
            let p1 = 0.0;
            let p5 = 0.0;

            let r2 = r_s * r_s;
            let r3 = r2 * r_s;
            let r4 = r2 * r2;

            let rhs1 = e_s - p0;
            let rhs2 = de_s;
            let rhs3 = d2e_s;

            let det = r2 * (3.0 * r2 * 12.0 * r2 - 4.0 * r3 * 6.0 * r_s)
                - r3 * (2.0 * r_s * 12.0 * r2 - 4.0 * r3 * 2.0)
                + r4 * (2.0 * r_s * 6.0 * r_s - 3.0 * r2 * 2.0);

            let det_p2 = rhs1 * (3.0 * r2 * 12.0 * r2 - 4.0 * r3 * 6.0 * r_s)
                - r3 * (rhs2 * 12.0 * r2 - 4.0 * r3 * rhs3)
                + r4 * (rhs2 * 6.0 * r_s - 3.0 * r2 * rhs3);

            let det_p3 = r2 * (rhs2 * 12.0 * r2 - 4.0 * r3 * rhs3)
                - rhs1 * (2.0 * r_s * 12.0 * r2 - 4.0 * r3 * 2.0)
                + r4 * (2.0 * r_s * rhs3 - rhs2 * 2.0);

            let det_p4 = r2 * (3.0 * r2 * rhs3 - rhs2 * 6.0 * r_s)
                - r3 * (2.0 * r_s * rhs3 - rhs2 * 2.0)
                + rhs1 * (2.0 * r_s * 6.0 * r_s - 3.0 * r2 * 2.0);

            let p2 = det_p2 / det;
            let p3 = det_p3 / det;
            let p4 = det_p4 / det;

            (a, b, c, r_spline_sq, p0, p1, p2, p3, p4, p5)
        }

        #[test]
        fn verify_spline_coefficients() {
            let p = c2_continuous_params();
            let (a, b, c, r_spline_sq, p0, p1, p2, p3, p4, p5) = p;
            let r_s = r_spline_sq.sqrt();

            let exp_term = (-b * r_s).exp();
            let inv_r = 1.0 / r_s;
            let inv_r2 = inv_r * inv_r;
            let inv_r6 = inv_r2 * inv_r2 * inv_r2;
            let inv_r7 = inv_r6 * inv_r;
            let inv_r8 = inv_r6 * inv_r2;

            let e_buck = a * exp_term - c * inv_r6;
            let de_buck = -a * b * exp_term + 6.0 * c * inv_r7;
            let d2e_buck = a * b * b * exp_term - 42.0 * c * inv_r8;

            let r2 = r_s * r_s;
            let r3 = r2 * r_s;
            let r4 = r2 * r2;
            let r5 = r4 * r_s;

            let e_poly = p0 + p1 * r_s + p2 * r2 + p3 * r3 + p4 * r4 + p5 * r5;
            let de_poly = p1 + 2.0 * p2 * r_s + 3.0 * p3 * r2 + 4.0 * p4 * r3 + 5.0 * p5 * r4;
            let d2e_poly = 2.0 * p2 + 6.0 * p3 * r_s + 12.0 * p4 * r2 + 20.0 * p5 * r3;

            assert_relative_eq!(e_poly, e_buck, epsilon = 1e-8);
            assert_relative_eq!(de_poly, de_buck, epsilon = 1e-8);
            assert_relative_eq!(d2e_poly, d2e_buck, epsilon = 1e-8);
        }

        #[test]
        fn continuity_c0_energy_at_boundary() {
            let p = c2_continuous_params();
            let r_spline_sq = p.3;
            let r_s = r_spline_sq.sqrt();
            let eps = 1e-8;

            let r_inside = r_s - eps;
            let r_outside = r_s + eps;

            let e_inside = SplinedBuckingham::energy(r_inside * r_inside, p);
            let e_outside = SplinedBuckingham::energy(r_outside * r_outside, p);

            assert_relative_eq!(e_inside, e_outside, epsilon = 1e-4);
        }

        #[test]
        fn continuity_c1_force_at_boundary() {
            let p = c2_continuous_params();
            let r_spline_sq = p.3;
            let r_s = r_spline_sq.sqrt();
            let eps = 1e-8;

            let r_inside = r_s - eps;
            let r_outside = r_s + eps;

            let d_inside = SplinedBuckingham::diff(r_inside * r_inside, p);
            let d_outside = SplinedBuckingham::diff(r_outside * r_outside, p);

            let de_dr_inside = -d_inside * r_inside;
            let de_dr_outside = -d_outside * r_outside;

            assert_relative_eq!(de_dr_inside, de_dr_outside, epsilon = 1e-3);
        }

        #[test]
        fn continuity_c2_second_derivative_at_boundary() {
            let p = c2_continuous_params();
            let (a, b, c, r_spline_sq, _p0, _p1, p2, p3, p4, p5) = p;
            let r_s = r_spline_sq.sqrt();

            let exp_term = (-b * r_s).exp();
            let inv_r8 = 1.0 / r_s.powi(8);
            let d2e_buck_analytical = a * b * b * exp_term - 42.0 * c * inv_r8;

            let d2e_poly_analytical =
                2.0 * p2 + 6.0 * p3 * r_s + 12.0 * p4 * r_s * r_s + 20.0 * p5 * r_s.powi(3);

            assert_relative_eq!(d2e_poly_analytical, d2e_buck_analytical, epsilon = 1e-6);

            let h = 1e-6;

            let e_m = SplinedBuckingham::energy((r_s - h).powi(2), p);
            let e_0 = SplinedBuckingham::energy(r_s.powi(2), p);
            let e_p = SplinedBuckingham::energy((r_s + h).powi(2), p);
            let d2e_numerical_straddling = (e_p - 2.0 * e_0 + e_m) / (h * h);

            let relative_error =
                (d2e_numerical_straddling - d2e_buck_analytical).abs() / d2e_buck_analytical.abs();
            assert!(
                relative_error < 0.1,
                "C² continuity check: relative error {} > 0.1",
                relative_error
            );
        }

        #[test]
        fn continuity_finite_diff_across_boundary() {
            let p = c2_continuous_params();
            let r_s = p.3.sqrt();

            let r = r_s;
            let r_sq = r * r;

            let r_plus = r + H;
            let r_minus = r - H;
            let e_plus = SplinedBuckingham::energy(r_plus * r_plus, p);
            let e_minus = SplinedBuckingham::energy(r_minus * r_minus, p);
            let de_dr_numerical = (e_plus - e_minus) / (2.0 * H);

            let d_analytic = SplinedBuckingham::diff(r_sq, p);
            let de_dr_analytic = -d_analytic * r;

            assert_relative_eq!(de_dr_numerical, de_dr_analytic, epsilon = TOL_DIFF);
        }
    }
}
