//! Non-bonded (intermolecular) interaction potentials.
//!
//! Non-bonded interactions occur between atoms not directly connected by the bond topology,
//! typically including van der Waals forces, electrostatics, and specific hydrogen bonding terms.
//!
//! # Components
//!
//! ## Van der Waals (vdW)
//! Models short-range repulsion (Pauli exclusion) and long-range attraction (dispersion).
//! - [`LennardJones`]: Classic 12-6 potential.
//! - [`Buckingham`]: Exponential-6 potential with softer repulsion.
//! - [`SplinedBuckingham`]: $C^2$ continuous variant of Buckingham with polynomial regularization at short range.
//!
//! ## Electrostatics
//! Models Coulombic interactions between partial charges.
//! - [`Coulomb`]: Standard $1/r$ electrostatic potential.
//!
//! ## Hydrogen Bonding
//! Explicit terms for hydrogen bond directionality and strength (DREIDING specific).
//! - [`HydrogenBond`]: 12-10 potential with angular dependence.

mod coulomb;
mod hbond;
mod vdw;

pub use coulomb::Coulomb;
pub use hbond::HydrogenBond;
pub use vdw::{Buckingham, LennardJones, SplinedBuckingham};
