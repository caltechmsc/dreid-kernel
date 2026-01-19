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
