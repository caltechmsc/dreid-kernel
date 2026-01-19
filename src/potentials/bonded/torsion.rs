use crate::math::Real;
use crate::traits::TorsionKernel;
use crate::types::EnergyDiff;

/// Periodic torsion potential for dihedral angles.
///
/// # Physics
///
/// Models the rotational barrier around a bond axis using a periodic cosine function.
///
/// - **Formula**: $$ E = \frac{1}{2} V [1 - \cos(n(\phi - \phi_0))] $$
/// - **Derivative (`diff`)**: $$ T = \frac{dE}{d\phi} = \frac{1}{2} V \cdot n \cdot \sin(n(\phi - \phi_0)) $$
///
/// # Parameters
///
/// - `v_half`: Half barrier height $V_{half} = V/2$.
/// - `n`: Periodicity/multiplicity.
/// - `cos_n_phi0`: $\cos(n\phi_0)$, pre-computed phase cosine.
/// - `sin_n_phi0`: $\sin(n\phi_0)$, pre-computed phase sine.
///
/// # Inputs
///
/// - `cos_phi`: Cosine of the dihedral angle $\cos\phi$.
/// - `sin_phi`: Sine of the dihedral angle $\sin\phi$.
///
/// # Implementation Notes
///
/// - Uses optimized closed-form formulas for common periodicities ($n = 1, 2, 3$).
/// - Falls back to Chebyshev recurrence for higher periodicities.
/// - All intermediate calculations are shared between energy and torque computations.
/// - Branchless and panic-free.
#[derive(Clone, Copy, Debug, Default)]
pub struct Torsion;

impl<T: Real> TorsionKernel<T> for Torsion {
    type Params = (T, u8, T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = V_{half} [1 - \cos(n(\phi - \phi_0))] $$
    #[inline(always)]
    fn energy(cos_phi: T, sin_phi: T, (v_half, n, cos_n_phi0, sin_n_phi0): Self::Params) -> T {
        let one = T::from(1.0f32);
        let (cos_n_phi, sin_n_phi) = multiple_angle(cos_phi, sin_phi, n);
        let cos_n_delta = cos_n_phi * cos_n_phi0 + sin_n_phi * sin_n_phi0;
        v_half * (one - cos_n_delta)
    }

    /// Computes only the torque $T$.
    ///
    /// # Formula
    ///
    /// $$ T = V_{half} \cdot n \cdot \sin(n(\phi - \phi_0)) $$
    ///
    /// This factor allows computing forces via the chain rule:
    /// $$ \vec{F} = -T \cdot \nabla \phi $$
    #[inline(always)]
    fn diff(cos_phi: T, sin_phi: T, (v_half, n, cos_n_phi0, sin_n_phi0): Self::Params) -> T {
        let (cos_n_phi, sin_n_phi) = multiple_angle(cos_phi, sin_phi, n);
        let sin_n_delta = sin_n_phi * cos_n_phi0 - cos_n_phi * sin_n_phi0;
        let n_t = T::from(n as f32);
        v_half * n_t * sin_n_delta
    }

    /// Computes both energy and torque efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(
        cos_phi: T,
        sin_phi: T,
        (v_half, n, cos_n_phi0, sin_n_phi0): Self::Params,
    ) -> EnergyDiff<T> {
        let one = T::from(1.0f32);

        let (cos_n_phi, sin_n_phi) = multiple_angle(cos_phi, sin_phi, n);

        let cos_n_delta = cos_n_phi * cos_n_phi0 + sin_n_phi * sin_n_phi0;
        let sin_n_delta = sin_n_phi * cos_n_phi0 - cos_n_phi * sin_n_phi0;

        let energy = v_half * (one - cos_n_delta);

        let n_t = T::from(n as f32);
        let diff = v_half * n_t * sin_n_delta;

        EnergyDiff { energy, diff }
    }
}

/// Computes $(\cos(n\phi), \sin(n\phi))$ using optimized paths for common $n$.
#[inline(always)]
fn multiple_angle<T: Real>(cos_phi: T, sin_phi: T, n: u8) -> (T, T) {
    match n {
        0 => multiple_angle_0(),
        1 => (cos_phi, sin_phi),
        2 => multiple_angle_2(cos_phi, sin_phi),
        3 => multiple_angle_3(cos_phi, sin_phi),
        _ => multiple_angle_chebyshev(cos_phi, sin_phi, n),
    }
}

/// $n = 0$: $(\cos(0), \sin(0)) = (1, 0)$.
#[inline(always)]
fn multiple_angle_0<T: Real>() -> (T, T) {
    (T::from(1.0f32), T::from(0.0f32))
}

/// $n = 2$: Double-angle formulas.
#[inline(always)]
fn multiple_angle_2<T: Real>(cos_phi: T, sin_phi: T) -> (T, T) {
    let one = T::from(1.0f32);
    let two = T::from(2.0f32);

    let cos_2phi = two * cos_phi * cos_phi - one;
    let sin_2phi = two * sin_phi * cos_phi;

    (cos_2phi, sin_2phi)
}

/// $n = 3$: Triple-angle formulas.
#[inline(always)]
fn multiple_angle_3<T: Real>(cos_phi: T, sin_phi: T) -> (T, T) {
    let three = T::from(3.0f32);
    let four = T::from(4.0f32);

    let cos2 = cos_phi * cos_phi;
    let sin2 = sin_phi * sin_phi;

    let cos_3phi = four * cos2 * cos_phi - three * cos_phi;
    let sin_3phi = three * sin_phi - four * sin2 * sin_phi;

    (cos_3phi, sin_3phi)
}

/// General case: Chebyshev recurrence for $n \geq 4$.
#[inline(always)]
fn multiple_angle_chebyshev<T: Real>(cos_phi: T, sin_phi: T, n: u8) -> (T, T) {
    let zero = T::from(0.0f32);
    let one = T::from(1.0f32);
    let two = T::from(2.0f32);

    let mut cos_prev2 = one;
    let mut sin_prev2 = zero;
    let mut cos_prev1 = cos_phi;
    let mut sin_prev1 = sin_phi;

    let two_cos = two * cos_phi;

    for _ in 2..=n {
        let cos_curr = two_cos * cos_prev1 - cos_prev2;
        let sin_curr = two_cos * sin_prev1 - sin_prev2;

        cos_prev2 = cos_prev1;
        sin_prev2 = sin_prev1;
        cos_prev1 = cos_curr;
        sin_prev1 = sin_curr;
    }

    (cos_prev1, sin_prev1)
}
