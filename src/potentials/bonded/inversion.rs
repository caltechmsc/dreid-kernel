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
