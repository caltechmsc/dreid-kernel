# Performance Benchmarks

This document details the performance metrics for `dreid-kernel`. All benchmarks were conducted using the `Criterion.rs` framework to ensure statistical significance.

## Executive Summary

The DREIDING force field kernel achieves extreme throughput by leveraging zero-cost abstractions and compiler optimizations.

- **Fastest Kernel**: Cosine Harmonic Angle (> 1.4 Billion Combined Ops/sec).
- **Most Common Kernel**: Lennard-Jones vdW (~830 Million Combined Ops/sec).
- **Complex Kernel**: Splined Buckingham (~110 Million Combined Ops/sec).
- **Optimization**: The `compute()` function (Energy + Force) is typically faster than the sum of their disparate calls due to shared intermediate calculations (e.g., $r^{-6}$ reuse).

## Detailed Results

**Legend:**

- **Energy**: Time to compute potential energy only ($E$).
- **Force**: Time to compute the force prefactor only ($-\frac{1}{r}\frac{dE}{dr}$).
- **Combined**: Time to compute both simultaneously. This utilizes shared sub-expressions (instruction-level optimization).
- **Throughput**: Based on the _Combined_ calculation time.

### Non-Bonded Interactions

| Potential              | Variant       | Energy (ns) | Force (ns) | Combined (ns) | Throughput (MOps/s) |
| :--------------------- | :------------ | :---------- | :--------- | :------------ | :------------------ |
| **Lennard-Jones**      | 12-6          | 0.94        | 0.97       | **1.19**      | **840**             |
| **Coulomb**            | Electrostatic | 2.08        | 2.07       | **2.18**      | **458**             |
| **Buckingham**         | Exp-6         | 5.77        | 7.25       | **7.73**      | **129**             |
| **Splined Buckingham** | Long Range    | 6.46        | 7.41       | **9.02**      | **110**             |
| **Splined Buckingham** | Short Range   | 6.46        | 7.62       | **8.78**      | **113**             |
| **Hydrogen Bond**      | 12-10 (n=4)   | 1.51        | 2.09       | **2.19**      | **456**             |

> _Note: Lennard-Jones combined calculation is extremely efficient, costing only ~0.25ns more than computing force alone._

### Bonded Interactions

| Group         | Potential           | Energy (ns) | Force (ns) | Combined (ns) | Throughput (MOps/s) |
| :------------ | :------------------ | :---------- | :--------- | :------------ | :------------------ |
| **Stretch**   | Harmonic            | 1.23        | 2.15       | **2.20**      | **454**             |
| **Stretch**   | Morse               | 4.61        | 5.92       | **6.25**      | **160**             |
| **Angle**     | **Cosine Harmonic** | **0.56**    | **0.55**   | **0.70**      | **1,428**           |
| **Angle**     | Theta Harmonic      | 5.02        | 5.84       | **6.14**      | **162**             |
| **Torsion**   | Cosine (n=1)        | 1.63        | 1.60       | **2.03**      | **492**             |
| **Torsion**   | Cosine (n=3)        | 2.13        | 2.18       | **2.55**      | **392**             |
| **Torsion**   | Cosine (n=6)        | 3.22        | 3.47       | **3.81**      | **262**             |
| **Inversion** | Planar              | 0.49        | 0.51       | **0.62**      | **1,612**           |
| **Inversion** | Umbrella            | 0.57        | 0.55       | **0.71**      | **1,408**           |

## Test Environment

- **CPU**: Intel® Core™ i7-13620H (Raptor Lake)
  - 10 Cores (6P + 4E), 16 Threads
  - Max Turbo Frequency: 4.90 GHz
  - Instruction Set: AVX2, FMA3
- **OS**: Linux (Arch Linux, Kernel 6.12.63-1)
- **Date**: January 2026

## Analysis

1. **Shared Instruction Optimization**: In almost all cases, `Combined < Energy + Force`. For example, in **Lennard-Jones**, computing both ($1.19\,\text{ns}$) is nearly as fast as computing just the force ($0.97\,\text{ns}$). This confirms that the compiler and CPU are effectively reusing the expensive $r^{-6}$ and $r^{-12}$ terms.
2. **Cosine vs Theta**: We observe a **10x speedup** when using the `CosineHarmonic` form ($\cos\theta - \cos\theta_0$) instead of the `ThetaHarmonic` form ($\theta - \theta_0$), due to the avoidance of the `acos` instruction.
3. **Branchless Spline**: The `SplinedBuckingham` kernel uses arithmetic masking (branchless logic) for the $r < r_{spline}$ check. By computing both paths and blending, it avoids expensive branch mispredictions, maintaining consistent throughput suitable for SIMD pipelines.
