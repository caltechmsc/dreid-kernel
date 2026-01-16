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

// ============================================================================
// Implementation for f32
// ============================================================================

impl Real for f32 {
    #[inline(always)]
    fn sqrt(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.sqrt()
        }
        #[cfg(not(feature = "std"))]
        {
            sqrtf(self)
        }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        self.recip()
    }

    #[inline(always)]
    fn abs(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.abs()
        }
        #[cfg(not(feature = "std"))]
        {
            fabsf(self)
        }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        #[cfg(feature = "std")]
        {
            self.max(other)
        }
        #[cfg(not(feature = "std"))]
        {
            if self > other { self } else { other }
        }
    }

    #[inline(always)]
    fn exp(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.exp()
        }
        #[cfg(not(feature = "std"))]
        {
            expf(self)
        }
    }

    #[inline(always)]
    fn sin(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.sin()
        }
        #[cfg(not(feature = "std"))]
        {
            sinf(self)
        }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.cos()
        }
        #[cfg(not(feature = "std"))]
        {
            cosf(self)
        }
    }

    #[inline(always)]
    fn acos(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.acos()
        }
        #[cfg(not(feature = "std"))]
        {
            acosf(self)
        }
    }
}

// ============================================================================
// Implementation for f64
// ============================================================================

impl Real for f64 {
    #[inline(always)]
    fn sqrt(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.sqrt()
        }
        #[cfg(not(feature = "std"))]
        {
            sqrt(self)
        }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        self.recip()
    }

    #[inline(always)]
    fn abs(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.abs()
        }
        #[cfg(not(feature = "std"))]
        {
            fabs(self)
        }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        #[cfg(feature = "std")]
        {
            self.max(other)
        }
        #[cfg(not(feature = "std"))]
        {
            if self > other { self } else { other }
        }
    }

    #[inline(always)]
    fn exp(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.exp()
        }
        #[cfg(not(feature = "std"))]
        {
            exp(self)
        }
    }

    #[inline(always)]
    fn sin(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.sin()
        }
        #[cfg(not(feature = "std"))]
        {
            sin(self)
        }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.cos()
        }
        #[cfg(not(feature = "std"))]
        {
            cos(self)
        }
    }

    #[inline(always)]
    fn acos(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.acos()
        }
        #[cfg(not(feature = "std"))]
        {
            acos(self)
        }
    }
}
