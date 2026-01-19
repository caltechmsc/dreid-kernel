//! Bonded (intramolecular) interaction potentials.
//!
//! Bonded interactions define the covalent structure and local flexibility of molecules.
//!
//! # Components
//!
//! ## 1. Bond Stretch (2-body)
//! Interactions between two directly bonded atoms.
//! - [`Harmonic`]: Standard harmonic spring potential.
//! - [`Morse`]: Anharmonic potential allowing bond dissociation.
//!
//! ## 2. Angle Bend (3-body)
//! Interactions defined by the angle between two bonds sharing a common atom.
//! - [`CosineHarmonic`]: Harmonic in cosine of the angle.
//! - [`ThetaHarmonic`]: Harmonic in the angle itself.
//!
//! ## 3. Torsion (4-body)
//! Interactions associated with rotation about a bond axis (dihedral angle).
//! - [`Torsion`]: Standard periodic cosine series expansion.
//!
//! ## 4. Inversion / Improper (4-body)
//! Interactions maintaining planarity or chiral centers.
//! - [`PlanarInversion`]: Harmonic potential on the out-of-plane distance.
//! - [`UmbrellaInversion`]: Harmonic potential on the Wilson angle.

mod angle;
mod inversion;
mod stretch;
mod torsion;

pub use angle::{CosineHarmonic, ThetaHarmonic};
pub use inversion::{PlanarInversion, UmbrellaInversion};
pub use stretch::{Harmonic, Morse};
pub use torsion::Torsion;
