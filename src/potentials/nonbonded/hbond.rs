use crate::math::Real;
use crate::traits::HybridKernel;
use crate::types::HybridEnergyDiff;

/// DREIDING Hydrogen Bond potential (12-10).
///
/// # Physics
///
/// Models the explicit hydrogen bonding interaction (typically D-H...A) using a
/// specific 12-10 Radial potential modulated by a $\cos^N\theta$ Angular term.
/// Standard DREIDING uses $N=4$.
///
/// - **Formula**: $$ E = D_{hb} \left[ 5 \left(\frac{R_{hb}}{r}\right)^{12} - 6 \left(\frac{R_{hb}}{r}\right)^{10} \right] \cos^N \theta $$
/// - **Derivative Factor (Radial)**: $$ D_{rad} = - \frac{1}{r} \frac{\partial E}{\partial r} = \frac{60 D_{hb}}{r^2} \left[ \left(\frac{R_{hb}}{r}\right)^{12} - \left(\frac{R_{hb}}{r}\right)^{10} \right] \cos^N \theta $$
/// - **Derivative Factor (Angular)**: $$ D_{ang} = \frac{\partial E}{\partial (\cos\theta)} = N \cdot E_{rad} \cos^{N-1} \theta $$
///
/// # Parameters
///
/// - `d_hb`: The energy well depth $D_{hb}$.
/// - `r_hb_sq`: The squared equilibrium distance $R_{hb}^2$.
/// - `N`: The cosine power exponent (const generic).
///
/// # Inputs
///
/// - `r_sq`: Squared distance $r^2$ between Donor (D) and Acceptor (A).
/// - `cos_theta`: Cosine of the angle $\theta_{DHA}$ (at Hydrogen).
///
/// # Implementation Notes
///
/// - **Cutoff**: If $\cos\theta \le 0$, energy and forces represent 0.
/// - **Optimization**: Uses $s = (R_{hb}/r)^2$ recurrence to compute $r^{-10}$ and $r^{-12}$ efficiently.
/// - **Generics**: Uses `const N: usize` to unroll power calculations at compile time.
#[derive(Clone, Copy, Debug, Default)]
pub struct HydrogenBond<const N: usize>;

impl<T: Real, const N: usize> HybridKernel<T> for HydrogenBond<N> {
    type Params = (T, T);

    /// Computes only the potential energy.
    ///
    /// # Formula
    ///
    /// $$ E = D_{hb} (5s^6 - 6s^5) \cos^N \theta, \quad \text{where } s = (R_{hb}/r)^2 $$
    #[inline(always)]
    fn energy(r_sq: T, cos_theta: T, (d_hb, r_hb_sq): Self::Params) -> T {
        let effective_cos = cos_theta.max(T::from(0.0));

        let cos_n = pow_n_helper(effective_cos, N);

        let s = r_hb_sq * r_sq.recip();
        let s2 = s * s;
        let s4 = s2 * s2;
        let s5 = s4 * s;
        let s6 = s4 * s2;

        let term12 = T::from(5.0) * s6;
        let term10 = T::from(6.0) * s5;

        (d_hb * (term12 - term10)) * cos_n
    }

    /// Computes only the derivative factors.
    ///
    /// # Formula
    ///
    /// $$ D_{rad} = \frac{60 D_{hb}}{r^2} (s^6 - s^5) \cos^N \theta, \quad \text{where } s = (R_{hb}/r)^2 $$
    /// $$ D_{ang} = N E_{rad} \cos^{N-1} \theta $$
    ///
    /// - `force_factor_rad` ($D_{rad}$): Used to compute the central force along the D-A axis:
    ///   $ \vec{F}_{rad} = -D\_{rad} \cdot \vec{r}\_{DA} $
    /// - `force_factor_ang` ($D_{ang}$): Used to compute torque-like forces on the D-H-A angle
    ///   via the Wilson B-matrix gradient chain rule:
    ///   $ \vec{F}_i = -D\_{ang} \cdot \nabla_i (\cos\theta) $
    #[inline(always)]
    fn diff(r_sq: T, cos_theta: T, (d_hb, r_hb_sq): Self::Params) -> (T, T) {
        let effective_cos = cos_theta.max(T::from(0.0));

        let inv_r2 = r_sq.recip();
        let s = r_hb_sq * inv_r2;
        let s2 = s * s;
        let s4 = s2 * s2;
        let s5 = s4 * s;
        let s6 = s4 * s2;

        let term12 = T::from(5.0) * s6;
        let term10 = T::from(6.0) * s5;
        let e_rad_pure = d_hb * (term12 - term10);

        let cos_n_minus_1 = if N == 0 {
            T::from(0.0)
        } else if N == 1 {
            T::from(1.0)
        } else {
            pow_n_helper(effective_cos, N - 1)
        };

        let cos_n = if N == 0 {
            T::from(1.0)
        } else {
            cos_n_minus_1 * effective_cos
        };

        let diff_rad_pure = T::from(60.0) * d_hb * inv_r2 * (s6 - s5);
        let force_factor_rad = diff_rad_pure * cos_n;

        let force_factor_ang = T::from(N as f32) * e_rad_pure * cos_n_minus_1;

        (force_factor_rad, force_factor_ang)
    }

    /// Computes both energy and derivative factors efficiently.
    ///
    /// This method reuses intermediate calculations to minimize operations.
    #[inline(always)]
    fn compute(r_sq: T, cos_theta: T, (d_hb, r_hb_sq): Self::Params) -> HybridEnergyDiff<T> {
        let effective_cos = cos_theta.max(T::from(0.0));

        let inv_r2 = r_sq.recip();
        let s = r_hb_sq * inv_r2;
        let s2 = s * s;
        let s4 = s2 * s2;
        let s5 = s4 * s;
        let s6 = s4 * s2;

        let term12 = T::from(5.0) * s6;
        let term10 = T::from(6.0) * s5;
        let e_rad_pure = d_hb * (term12 - term10);

        let cos_n_minus_1 = if N == 0 {
            T::from(0.0)
        } else if N == 1 {
            T::from(1.0)
        } else {
            pow_n_helper(effective_cos, N - 1)
        };

        let cos_n = if N == 0 {
            T::from(1.0)
        } else {
            cos_n_minus_1 * effective_cos
        };

        let energy = e_rad_pure * cos_n;

        let diff_rad_pure = T::from(60.0) * d_hb * inv_r2 * (s6 - s5);
        let force_factor_rad = diff_rad_pure * cos_n;

        let force_factor_ang = T::from(N as f32) * e_rad_pure * cos_n_minus_1;

        HybridEnergyDiff {
            energy,
            force_factor_rad,
            force_factor_ang,
        }
    }
}

/// Helper to compute x^n using explicit unrolling for small common powers,
/// and fast exponentiation for larger n.
#[inline(always)]
fn pow_n_helper<T: Real>(base: T, n: usize) -> T {
    match n {
        0 => T::from(1.0),
        1 => base,
        2 => base * base,
        3 => base * base * base,
        4 => {
            let x2 = base * base;
            x2 * x2
        }
        5 => {
            let x2 = base * base;
            let x4 = x2 * x2;
            x4 * base
        }
        6 => {
            let x2 = base * base;
            let x4 = x2 * x2;
            x4 * x2
        }
        _ => {
            let mut acc = T::from(1.0);
            let mut b = base;
            let mut e = n;
            while e > 0 {
                if e & 1 == 1 {
                    acc = acc * b;
                }
                b = b * b;
                e >>= 1;
            }
            acc
        }
    }
}
