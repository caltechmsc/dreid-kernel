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
    /// by a single vector multiplication: $\vec{F} = D \cdot \vec{r}$.
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
    /// by a single vector multiplication: $\vec{F} = D \cdot \vec{r}$.
    #[inline(always)]
    fn diff(r_sq: T, (a, b, c, r_fusion_sq): Self::Params) -> T {
        let is_safe = T::from((r_sq >= r_fusion_sq) as u8 as f32);
        let effective_r_sq = r_sq.max(r_fusion_sq);

        let r = effective_r_sq.sqrt();
        let inv_r2 = effective_r_sq.recip();
        let inv_r4 = inv_r2 * inv_r2;
        let inv_r8 = inv_r4 * inv_r4;

        let exp_term = T::exp(-b * r);

        let repulsion_factor = a * b * exp_term * r.recip();
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

        let r = effective_r_sq.sqrt();
        let inv_r2 = effective_r_sq.recip();
        let inv_r4 = inv_r2 * inv_r2;
        let inv_r6 = inv_r4 * inv_r2;
        let inv_r8 = inv_r6 * inv_r2;

        let exp_term = T::exp(-b * r);

        let repulsion_energy = a * exp_term;
        let attraction_energy = c * inv_r6;
        let energy_unsafe = repulsion_energy - attraction_energy;

        let repulsion_force = repulsion_energy * b * r.recip();
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
