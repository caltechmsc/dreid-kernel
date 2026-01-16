#[cfg(not(feature = "std"))]
use libm::{acos, acosf, cos, cosf, exp, expf, fabs, fabsf, sin, sinf, sqrt, sqrtf};

/// Abstract trait for floating-point operations required by force field kernels.
///
/// This trait abstracts over `f32` and `f64`, providing a unified interface
/// for mathematical functions whether in `std` or `no_std` environments.
pub trait Real:
    Copy
    + Clone
    + PartialOrd
    + PartialEq
    + core::fmt::Debug
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
    + From<f32>
{
    // Basic functions
    fn sqrt(self) -> Self;
    fn recip(self) -> Self;
    fn abs(self) -> Self;
    fn max(self, other: Self) -> Self;

    // Transcendental functions
    fn exp(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn acos(self) -> Self;

    // Composite helpers (can be specialized by hardware)
    #[inline(always)]
    fn rsqrt(self) -> Self {
        self.sqrt().recip()
    }
}
