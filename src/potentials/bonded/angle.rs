use crate::math::Real;
use crate::traits::AngleKernel;
use crate::types::EnergyDiff;

/// Cosine Harmonic potential for bond angles.
///
/// # Physics
///
/// Models the angle bending energy using a simple harmonic approximation on the cosine of the angle.
///
/// - **Formula**: $$ E = \frac{1}{2} C (\cos\theta - \cos\theta_0)^2 $$
/// - **Derivative Factor (`diff`)**: $$ \Gamma = \frac{dE}{d(\cos\theta)} = C (\cos\theta - \cos\theta_0) $$
///
/// # Parameters
///
/// - `k_half`: Half force constant $C_{half} = C/2$.
/// - `cos0`: The cosine of the equilibrium angle $\cos\theta_0$.
///
/// # Inputs
///
/// - `cos_theta`: Cosine of the bond angle $\theta_{ijk}$.
///
/// # Implementation Notes
///
/// - Pure polynomial evaluation; no `acos`, `sin`, or `sqrt` required.
/// - All intermediate calculations are shared between energy and force computations.
/// - Branchless and panic-free.
#[derive(Clone, Copy, Debug, Default)]
pub struct CosineHarmonic;

impl<T: Real> AngleKernel<T> for CosineHarmonic {
    type Params = (T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = C_{half} (\Delta)^2, \quad \text{where } \Delta = \cos\theta - \cos\theta_0 $$
    #[inline(always)]
    fn energy(cos_theta: T, (k_half, cos0): Self::Params) -> T {
        let delta = cos_theta - cos0;
        k_half * delta * delta
    }

    /// Computes only the derivative factor $\Gamma$.
    ///
    /// # Formula
    ///
    /// $$ \Gamma = 2 C_{half} (\cos\theta - \cos\theta_0) $$
    ///
    /// This factor allows computing forces via the chain rule:
    /// $$ \vec{F} = -\Gamma \cdot \nabla (\cos\theta) $$
    #[inline(always)]
    fn diff(cos_theta: T, (k_half, cos0): Self::Params) -> T {
        let c = k_half + k_half;
        c * (cos_theta - cos0)
    }

    /// Computes both energy and derivative factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(cos_theta: T, (k_half, cos0): Self::Params) -> EnergyDiff<T> {
        let delta = cos_theta - cos0;

        let energy = k_half * delta * delta;

        let c = k_half + k_half;
        let diff = c * delta;

        EnergyDiff { energy, diff }
    }
}
