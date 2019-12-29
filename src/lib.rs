//! mlearn: Machine Learning Tools in Rust
//!
//! This package is meant to provide various machine learning tools
//! implemented in Rust.

// Basic tools
pub mod metrics;

// Regression
pub mod regression;
pub mod linear_regression;

// Classification
pub mod classification;

// High level model interfaces
pub mod model_builder;
