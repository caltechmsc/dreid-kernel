use crate::math::Real;
use crate::types::{EnergyDiff, HybridEnergyDiff};

// ============================================================================
// 1. Pair Potential (2-Body Interaction)
// ============================================================================

/// Trait for 2-body potentials (Bond Stretch, Van der Waals, Coulomb).
///
/// Models interaction energy as a function of squared distance $r^2$.
///
/// # Mathematical Contract
/// - **Input**: Squared distance $r^2$ (to avoid unnecessary square roots in cutoff checks).
/// - **Output (Diff)**: The radial force pre-factor $D$ defined as:
///   $$ D = -\frac{1}{r} \frac{dE}{dr} $$
pub trait PairKernel<T: Real> {
    /// Associated constants/parameters required by the potential (e.g., $k, r_0$).
    type Params: Copy;

    /// Computes only the potential energy.
    fn energy(r_sq: T, params: Self::Params) -> T;

    /// Computes only the force pre-factor $D$.
    fn diff(r_sq: T, params: Self::Params) -> T;

    /// Computes both energy and force pre-factor efficiently.
    /// Should share intermediate calculations (e.g., `sqrt`, `exp`) where possible.
    fn compute(r_sq: T, params: Self::Params) -> EnergyDiff<T>;
}
