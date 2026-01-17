use crate::math::Real;

/// Standard output for most potentials (2-body, 3-body, 4-body).
///
/// Contains the scalar potential energy and a single derivative factor
/// used to compute forces or torques in the upper geometry layer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EnergyDiff<T: Real> {
    pub energy: T,
    pub diff: T,
}

/// Specialized output for Hybrid potentials (e.g., Hydrogen Bonds).
///
/// Hybrid potentials depend on both distance (r) and angle (theta),
/// producing two distinct derivative factors for force distribution.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HybridEnergyDiff<T: Real> {
    pub energy: T,

    /// Radial force factor: `-(1/r * dE/dr)`
    /// Used to compute forces along the D-A vector.
    pub force_factor_rad: T,

    /// Angular force factor: `-dE/d(cos)`
    /// Used to compute torque-like forces on D-H-A via Wilson B-Matrix.
    pub force_factor_ang: T,
}
