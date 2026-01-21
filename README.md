# DREID-Kernel

**A high-performance, no-std Rust library providing pure mathematical primitives and stateless energy kernels for the DREIDING force field.**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Verification](#verification) • [Benchmarks](#performance) • [License](#license)

---

## Features

- **⚡ Bare Metal Ready**: Strict `#![no_std]` support. Zero heap allocations. Use this in embedded devices, OS kernels, or WASM environments.
- **📐 Stateless Design**: Pure mathematical functions. You provide the state (coordinates/parameters), we return the forces. No object lifecycle management.
- **🔬 DREIDING Compliant**: Implements the exact functional forms defined in Mayo et al. (1990), including specific hydrogen bonding and inversion terms.
- **🛡️ Type Safe**: Uses Rust's trait system to enforce compile-time correctness for potential parameters and inputs.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
dreid-kernel = "0.1.0"
```

## Usage

This library provides low-level kernels. It is designed to be the calculation engine for higher-level molecular dynamics software.

### Example: Van der Waals Interaction

```rust
use dreid_kernel::{potentials::nonbonded::LennardJones, PairKernel};

// Constants (compile-time or runtime)
// (Depth, Equilibrium Distance^2)
let params = (0.1, 16.0);

// Squared distance between atoms (r^2 = 3.8^2)
let r_sq = 14.44;

// Compute Energy
let energy = LennardJones::energy(r_sq, params);

// Compute Force Prefactor (-1/r * dE/dr)
let diff = LennardJones::diff(r_sq, params);

// Compute Both (Optimized)
let (e, d) = LennardJones::compute(r_sq, params);
```

### Example: Torsion Angle

```rust
use dreid_kernel::{potentials::bonded::Torsion, TorsionKernel};

// (V_half, Periodicity n, Cos(n*phase), Sin(n*phase))
// V_half = V/2 where V is the full barrier height
let params = (2.5, 3, 1.0, 0.0); // V=5 kcal/mol, n=3, phase=0

// Dihedral angle input (cos(phi), sin(phi))
let cos_phi = 0.5;
let sin_phi = 0.866025;

let energy = Torsion::energy(cos_phi, sin_phi, params);
```

> More usage instructions and examples can be found in the [documentation](https://docs.rs/dreid-kernel).

## Verification

This library enforces a "Zero Technical Debt" policy. Every kernel is verified by a "Trinity of Tests" before release:

1. **Analytical Consistency**: We mathematically verify that the implemented force function is exactly the negative gradient of the energy function ($F = -\nabla E$).
2. **Finite Difference Validation**: We numerically prove (using Machine Epsilon steps) that the analytic derivatives match the numerical slope of the energy surface to within $10^{-6}$ precision.
3. **Physical Regression**: We explicitly test equilibrium conditions and asymptotic behavior (e.g., dispersion attraction at long range, Pauli repulsion at short range).

Run the full verification suite:

```bash
cargo test
```

## Architecture

The force calculation follows a two-layer design:

1. **Kernel Layer** (this crate): Computes scalar energy and derivative factors.
2. **Geometry Layer** (your code): Applies `F += -Factor * GeometricVector`.

This separation allows the kernel to be completely geometry-agnostic,
enabling use in periodic boundary conditions, minimization, or MD without modification.

## Performance

We benchmark every kernel using rigorous statistical sampling (via `criterion`). Benchmarks prevent compiler optimizations from eliding calculations (`black_box`) to measure true instruction throughput.

**Summary of Results (Intel Core i7-13620H) - Single Threaded:**

| Interaction  | Kernel           | Combined Time | Throughput     |
| :----------- | :--------------- | :------------ | :------------- |
| **Angle**    | Cosine Harmonic  | **0.70 ns**   | ~1.4 Billion/s |
| **vdW**      | Lennard-Jones    | **1.19 ns**   | ~840 Million/s |
| **Stretch**  | Harmonic Stretch | **2.20 ns**   | ~450 Million/s |
| **Dihedral** | Torsion (n=3)    | **2.55 ns**   | ~390 Million/s |

> For the complete data set, including hardware specifications, methodology, and analysis of `exp()` vs polynomial costs, see [**BENCHMARKS.md**](BENCHMARKS.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for the scientific computing community**
