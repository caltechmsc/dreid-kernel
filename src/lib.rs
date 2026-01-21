#![cfg_attr(not(feature = "std"), no_std)]

//! # DREID-Kernel
//!
//! **A high-performance, no-std Rust library providing pure mathematical primitives
//! and stateless energy kernels for the DREIDING force field.**
//!
//! ## Features
//!
//! - **⚡ Bare Metal Ready**: Strict `#![no_std]` support. Zero heap allocations.
//!   Use this in embedded devices, OS kernels, or WASM environments.
//! - **📐 Stateless Design**: Pure mathematical functions. You provide the state
//!   (coordinates/parameters), we return the forces. No object lifecycle management.
//! - **🔬 DREIDING Compliant**: Implements the exact functional forms defined in
//!   Mayo et al. (1990), including specific hydrogen bonding and inversion terms.
//! - **🛡️ Type Safe**: Uses Rust's trait system to enforce compile-time correctness
//!   for potential parameters and inputs.
//!
//! ## Kernel Traits
//!
//! This library provides four kernel trait families:
//!
//! | Trait | Body | Input | Output |
//! |-------|------|-------|--------|
//! | [`PairKernel`] | 2-body | $r^2$ | $(E, D)$ where $D = -\frac{1}{r}\frac{dE}{dr}$ |
//! | [`AngleKernel`] | 3-body | $\cos\theta$ | $(E, \Gamma)$ where $\Gamma = \frac{dE}{d(\cos\theta)}$ |
//! | [`TorsionKernel`] | 4-body | $(\cos\phi, \sin\phi)$ | $(E, T)$ where $T = \frac{dE}{d\phi}$ |
//! | [`HybridKernel`] | Mixed | $(r^2, \cos\theta)$ | $(E, D_{rad}, D_{ang})$ |
//!
//! ## Available Potentials
//!
//! ### Non-Bonded Interactions ([`potentials::nonbonded`])
//!
//! | Category | Kernel | Description |
//! |----------|--------|-------------|
//! | Van der Waals | [`LennardJones`](potentials::nonbonded::LennardJones) | Classic 12-6 potential |
//! | Van der Waals | [`Buckingham`](potentials::nonbonded::Buckingham) | Exponential-6 potential |
//! | Van der Waals | [`SplinedBuckingham`](potentials::nonbonded::SplinedBuckingham) | $C^2$ continuous Exp-6 |
//! | Electrostatics | [`Coulomb`](potentials::nonbonded::Coulomb) | Standard $1/r$ potential |
//! | H-Bond | [`HydrogenBond`](potentials::nonbonded::HydrogenBond) | 12-10 with angular term |
//!
//! ### Bonded Interactions ([`potentials::bonded`])
//!
//! | Category | Kernel | Description |
//! |----------|--------|-------------|
//! | Stretch | [`Harmonic`](potentials::bonded::Harmonic) | Harmonic spring |
//! | Stretch | [`Morse`](potentials::bonded::Morse) | Anharmonic with dissociation |
//! | Angle | [`CosineHarmonic`](potentials::bonded::CosineHarmonic) | Harmonic in $\cos\theta$ |
//! | Angle | [`ThetaHarmonic`](potentials::bonded::ThetaHarmonic) | Harmonic in $\theta$ |
//! | Torsion | [`Torsion`](potentials::bonded::Torsion) | Periodic cosine series |
//! | Inversion | [`PlanarInversion`](potentials::bonded::PlanarInversion) | Planar constraint |
//! | Inversion | [`UmbrellaInversion`](potentials::bonded::UmbrellaInversion) | Umbrella constraint |
//!
//! ## Quick Start
//!
//! ### Van der Waals Interaction
//!
//! ```
//! use dreid_kernel::{potentials::nonbonded::LennardJones, PairKernel};
//!
//! // Parameters: (D0, R0^2) - well depth and equilibrium distance squared
//! let params = (0.1, 16.0);
//!
//! // Squared distance between atoms
//! let r_sq = 14.44;
//!
//! // Compute energy
//! let energy = LennardJones::energy(r_sq, params);
//!
//! // Compute force prefactor: D = -(1/r) * dE/dr
//! let diff = LennardJones::diff(r_sq, params);
//!
//! // Compute both (optimized, shares intermediate calculations)
//! let result = LennardJones::compute(r_sq, params);
//! ```
//!
//! ### Torsion Angle
//!
//! ```
//! use dreid_kernel::{potentials::bonded::Torsion, TorsionKernel};
//!
//! // Parameters: (V_half, n, cos(n*phi0), sin(n*phi0))
//! // V_half = V/2 where V is barrier height
//! let params = (2.5, 3, 1.0, 0.0); // V=5 kcal/mol, n=3, phase=0
//!
//! // Dihedral angle as (cos(phi), sin(phi))
//! let cos_phi = 0.5;
//! let sin_phi = 0.866025;
//!
//! let energy = Torsion::energy(cos_phi, sin_phi, params);
//! ```
//!
//! ## Architecture
//!
//! The force calculation follows a two-layer design:
//!
//! 1. **Kernel Layer** (this crate): Computes scalar energy and derivative factors.
//! 2. **Geometry Layer** (your code): Applies `F += -Factor * GeometricVector`.
//!
//! This separation allows the kernel to be completely geometry-agnostic,
//! enabling use in periodic boundary conditions, minimization, or MD without modification.
//!
//! ## Performance
//!
//! Benchmarked on Intel Core i7-13620H (Single Threaded):
//!
//! | Kernel | Combined Time | Throughput |
//! |--------|---------------|------------|
//! | Cosine Harmonic | 0.70 ns | ~1.4 Billion ops/sec |
//! | Lennard-Jones | 1.19 ns | ~840 Million ops/sec |
//! | Torsion (n=3) | 2.55 ns | ~390 Million ops/sec |
//!
//! See [`BENCHMARKS.md`](https://github.com/caltechmsc/dreid-kernel/blob/main/BENCHMARKS.md) for complete data.
//!
//! **Made with ❤️ for the scientific computing community**

mod math;
mod traits;
mod types;

pub mod potentials;

pub use math::Real;

pub use traits::{AngleKernel, HybridKernel, PairKernel, TorsionKernel};
pub use types::{EnergyDiff, HybridEnergyDiff};
