//! Molecular mechanics potential energy functions.
//!
//! This module organizes the physical interaction kernels used in the DREIDING force field.
//! These kernels are pure functions that compute potential energy and force derivatives
//! based on geometric parameters (distances, angles) and force field constants.
//!
//! # Categories
//!
//! - [`bonded`]: Intramolecular interactions (covalent bonds, angles, dihedrals).
//! - [`nonbonded`]: Intermolecular and non-bonded intramolecular interactions (VDW, Electrostatics).

pub mod bonded;
pub mod nonbonded;
