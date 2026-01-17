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

// ============================================================================
// 2. Angle Potential (3-Body / 4-Body Planar Interaction)
// ============================================================================

/// Trait for bending potentials (Angle) and inversion potentials (Improper).
///
/// Models interaction energy as a function of the cosine of an angle $\theta$ or $\psi$.
///
/// # Mathematical Contract
/// - **Input**: Cosine of the angle ($\cos\theta$).
///   - For Angles: $\cos\theta = \hat{r}\_{ji} \cdot \hat{r}\_{jk}$
///   - For Inversions: $\cos\psi = \hat{n}\_{jik} \cdot \hat{r}\_{il}$
/// - **Output (Diff)**: The torque-like factor $\Gamma$ defined as:
///   $$ \Gamma = -\frac{dE}{d(\cos\theta)} $$
pub trait AngleKernel<T: Real> {
    /// Associated constants/parameters required by the potential (e.g., $k, \theta_0$).
    type Params: Copy;

    /// Computes only the potential energy.
    fn energy(cos_angle: T, params: Self::Params) -> T;

    /// Computes only the torque-like factor $\Gamma$.
    fn diff(cos_angle: T, params: Self::Params) -> T;

    /// Computes both energy and torque-like factor efficiently.
    /// Should share intermediate calculations where possible.
    fn compute(cos_angle: T, params: Self::Params) -> EnergyDiff<T>;
}

// ============================================================================
// 3. Torsion Potential (4-Body Interaction)
// ============================================================================

/// Trait for torsional potentials (Dihedrals).
///
/// Models interaction energy as a function of the dihedral angle $\phi$.
///
/// # Mathematical Contract
/// - **Input**: Both $\cos\phi$ and $\sin\phi$ are required to determine phase and
///   compute multi-term expansions (e.g., $\cos(n\phi)$) without `acos`.
/// - **Output (Diff)**: The pure torque $T$ defined as:
///   $$ T = -\frac{dE}{d\phi} $$
pub trait TorsionKernel<T: Real> {
    /// Associated constants/parameters required by the potential (e.g., $V, n$).
    type Params: Copy;

    /// Computes only the potential energy.
    fn energy(cos_phi: T, sin_phi: T, params: Self::Params) -> T;

    /// Computes only the pure torque $T$.
    fn diff(cos_phi: T, sin_phi: T, params: Self::Params) -> T;

    /// Computes both energy and pure torque efficiently.
    /// Should share intermediate calculations where possible.
    fn compute(cos_phi: T, sin_phi: T, params: Self::Params) -> EnergyDiff<T>;
}

// ============================================================================
// 4. Hybrid Potential (Mixed Interaction)
// ============================================================================

/// Trait for potentials dependent on both distance and angle (e.g., H-Bonds).
///
/// Models interaction energy as a function of squared distance $r^2$ and
/// cosine of angle $\cos\theta$.
///
/// # Mathematical Contract
/// - **Input**: Squared distance $r^2$ and cosine of angle $\cos\theta$.
/// - **Output**: Two derivative factors:
///   1. `force_factor_rad`: Radial part ($1/r \cdot dE/dr$).
///   2. `force_factor_ang`: Angular part ($-dE/d\cos$).
pub trait HybridKernel<T: Real> {
    /// Associated constants/parameters required by the potential.
    type Params: Copy;

    /// Computes only the potential energy.
    fn energy(r_sq: T, cos_theta: T, params: Self::Params) -> T;

    /// Computes only the derivative factors.
    fn diff(r_sq: T, cos_theta: T, params: Self::Params) -> HybridEnergyDiff<T>;

    /// Computes both energy and derivative factors efficiently.
    /// Should share intermediate calculations where possible.
    fn compute(r_sq: T, cos_theta: T, params: Self::Params) -> HybridEnergyDiff<T>;
}
