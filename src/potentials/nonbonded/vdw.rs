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
