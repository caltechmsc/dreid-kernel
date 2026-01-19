use crate::math::Real;
use crate::traits::PairKernel;
use crate::types::EnergyDiff;

/// Harmonic potential implementation for 1-2 bond stretching.
///
/// # Physics
///
/// Models the bond stretching as a harmonic oscillator.
///
/// - **Formula**: $$ E = \frac{1}{2} K (R - R_0)^2 $$
/// - **Derivative Factor (`diff`)**: $$ D = -\frac{K (R - R_0)}{R} $$
///
/// # Parameters
///
/// - `k_half`: Half force constant $K_{half} = K/2$.
/// - `r0`: Equilibrium distance $R_0$.
///
/// # Inputs
///
/// - `r_sq`: Squared distance $r^2$ between two atoms.
///
/// # Implementation Notes
///
/// - Requires square root to obtain $R$.
/// - All intermediate calculations are shared between energy and force computations.
/// - Branchless and panic-free.
#[derive(Clone, Copy, Debug, Default)]
pub struct Harmonic;

impl<T: Real> PairKernel<T> for Harmonic {
    type Params = (T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = K_{half} (R - R_0)^2 $$
    #[inline(always)]
    fn energy(r_sq: T, (k_half, r0): Self::Params) -> T {
        let r = r_sq.sqrt();
        let delta = r - r0;
        k_half * delta * delta
    }

    /// Computes only the force pre-factor $D$.
    ///
    /// # Formula
    ///
    /// $$ D = -\frac{2 K_{half} (R - R_0)}{R} $$
    ///
    /// This factor is defined such that the force vector can be computed
    /// by a single vector multiplication: $\vec{F} = -D \cdot \vec{r}$.
    #[inline(always)]
    fn diff(r_sq: T, (k_half, r0): Self::Params) -> T {
        let inv_r = r_sq.rsqrt();
        let r = r_sq * inv_r;
        let delta = r - r0;

        let k = k_half + k_half;

        -k * delta * inv_r
    }

    /// Computes both energy and force pre-factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(r_sq: T, (k_half, r0): Self::Params) -> EnergyDiff<T> {
        let inv_r = r_sq.rsqrt();
        let r = r_sq * inv_r;
        let delta = r - r0;

        let energy = k_half * delta * delta;

        let k = k_half + k_half;
        let diff = -k * delta * inv_r;

        EnergyDiff { energy, diff }
    }
}

/// Morse potential implementation for 1-2 bond stretching.
///
/// # Physics
///
/// Models bond stretching with anharmonicity, allowing for bond dissociation.
///
/// - **Formula**: $$ E = D_e [ e^{-\alpha(R - R_0)} - 1 ]^2 $$
/// - **Derivative Factor (`diff`)**: $$ D = \frac{2 \alpha D_e e^{-\alpha(R - R_0)} \left( e^{-\alpha(R - R_0)} - 1 \right)}{R} $$
///
/// # Parameters
///
/// - `de`: Dissociation energy $D_e$.
/// - `r0`: Equilibrium distance $R_0$.
/// - `alpha`: Stiffness parameter $\alpha$.
///
/// # Inputs
///
/// - `r_sq`: Squared distance $r^2$ between two atoms.
///
/// # Implementation Notes
///
/// - Requires `sqrt` and `exp`.
/// - More computationally expensive than Harmonic.
#[derive(Clone, Copy, Debug, Default)]
pub struct Morse;

impl<T: Real> PairKernel<T> for Morse {
    type Params = (T, T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = D_e [ e^{-\alpha(R - R_0)} - 1 ]^2 $$
    #[inline(always)]
    fn energy(r_sq: T, (de, r0, alpha): Self::Params) -> T {
        let r = r_sq.sqrt();
        let t_val = T::exp(-alpha * (r - r0));
        let term = t_val - T::from(1.0);

        de * term * term
    }

    /// Computes only the force pre-factor $D$.
    ///
    /// # Formula
    ///
    /// $$ D = \frac{2 \alpha D_e e^{-\alpha(R - R_0)} \left( e^{-\alpha(R - R_0)} - 1 \right)}{R} $$
    ///
    /// This factor is defined such that the force vector can be computed
    /// by a single vector multiplication: $\vec{F} = -D \cdot \vec{r}$.
    #[inline(always)]
    fn diff(r_sq: T, (de, r0, alpha): Self::Params) -> T {
        let inv_r = r_sq.rsqrt();
        let r = r_sq * inv_r;

        let t_val = T::exp(-alpha * (r - r0));
        let term_minus_one = t_val - T::from(1.0);

        let f_mag = T::from(2.0) * alpha * de * t_val * term_minus_one;

        f_mag * inv_r
    }

    /// Computes both energy and force pre-factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(r_sq: T, (de, r0, alpha): Self::Params) -> EnergyDiff<T> {
        let inv_r = r_sq.rsqrt();
        let r = r_sq * inv_r;

        let t_val = T::exp(-alpha * (r - r0));
        let term_minus_one = t_val - T::from(1.0);

        let energy = de * term_minus_one * term_minus_one;

        let f_mag = T::from(2.0) * alpha * de * t_val * term_minus_one;
        let diff = f_mag * inv_r;

        EnergyDiff { energy, diff }
    }
}
