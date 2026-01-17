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
