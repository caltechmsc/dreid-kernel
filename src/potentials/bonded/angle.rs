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

/// Theta Harmonic potential for bond angles.
///
/// # Physics
///
/// Models the angle bending energy using a harmonic approximation directly on the angle $\theta$ (in radians).
///
/// - **Formula**: $$ E = \frac{1}{2} K (\theta - \theta_0)^2 $$
/// - **Derivative Factor (`diff`)**: $$ \Gamma = \frac{dE}{d(\cos\theta)} = -K \frac{\theta - \theta_0}{\sin\theta} $$
///
/// # Parameters
///
/// - `k_half`: Half force constant $K_{half} = K/2$.
/// - `theta0`: The equilibrium angle $\theta_0$ in radians.
///
/// # Inputs
///
/// - `cos_theta`: Cosine of the bond angle $\theta_{ijk}$.
///
/// # Implementation Notes
///
/// - Uses `k_half` to save one multiplication in the energy step.
/// - Handles $\theta=0$ and $\theta=\pi$ analytically using L'Hopital's rule.
/// - Needs a single `acos` call for angle calculation.
#[derive(Clone, Copy, Debug, Default)]
pub struct ThetaHarmonic;

impl<T: Real> AngleKernel<T> for ThetaHarmonic {
    type Params = (T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = K_{half} (\theta - \theta_0)^2 $$
    #[inline(always)]
    fn energy(cos_theta: T, (k_half, theta0): Self::Params) -> T {
        let one = T::from(1.0f32);
        let minus_one = T::from(-1.0f32);

        let c = cos_theta.max(minus_one).min(one);
        let theta = c.acos();

        let delta = theta - theta0;

        k_half * delta * delta
    }

    /// Computes only the derivative factor $\Gamma$.
    ///
    /// # Formula
    ///
    /// $$ \Gamma = -2 K_{half} \frac{\theta - \theta_0}{\sin\theta} $$
    ///
    /// This factor allows computing forces via the chain rule:
    /// $$ \vec{F} = -\Gamma \cdot \nabla (\cos\theta) $$
    #[inline(always)]
    fn diff(cos_theta: T, (k_half, theta0): Self::Params) -> T {
        let one = T::from(1.0f32);
        let minus_one = T::from(-1.0f32);
        let zero = T::from(0.0f32);
        let singularity_thresh = T::from(1.0e-4f32);
        let epsilon = T::from(1.0e-20f32);

        let c = cos_theta.max(minus_one).min(one);

        let theta = c.acos();
        let sin_theta = (one - c * c).max(zero).sqrt();

        let factor = if sin_theta > singularity_thresh {
            (theta - theta0) / sin_theta
        } else {
            let s_safe = sin_theta.max(epsilon);

            if c > zero {
                one - theta0 / s_safe
            } else {
                let pi = T::pi();
                minus_one + (pi - theta0) / s_safe
            }
        };

        let k = k_half + k_half;

        -k * factor
    }

    /// Computes both energy and derivative factor efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(cos_theta: T, (k_half, theta0): Self::Params) -> EnergyDiff<T> {
        let one = T::from(1.0f32);
        let minus_one = T::from(-1.0f32);
        let zero = T::from(0.0f32);
        let singularity_thresh = T::from(1.0e-4f32);
        let epsilon = T::from(1.0e-20f32);

        let c = cos_theta.max(minus_one).min(one);
        let theta = c.acos();
        let sin_theta = (one - c * c).max(zero).sqrt();

        let delta = theta - theta0;
        let energy = k_half * delta * delta;

        let factor = if sin_theta > singularity_thresh {
            delta / sin_theta
        } else {
            let s_safe = sin_theta.max(epsilon);
            if c > zero {
                one - theta0 / s_safe
            } else {
                let pi = T::pi();
                minus_one + (pi - theta0) / s_safe
            }
        };

        let k = k_half + k_half;
        let diff = -k * factor;

        EnergyDiff { energy, diff }
    }
}
