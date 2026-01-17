#![cfg_attr(not(feature = "std"), no_std)]

pub mod math;
mod types;

pub use types::{EnergyDiff, HybridEnergyDiff};
