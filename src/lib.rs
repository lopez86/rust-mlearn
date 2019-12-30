//! # mlearn: Machine Learning Tools in Rust
//!
//! This package is meant to provide various machine learning tools
//! implemented in Rust.
//!
//! ## Example: Running a linear regression.
//!
//! ```.rust
//! extern crate ndarray;
//! extern crate netlib_src;
//!
//! use ndarray::array;
//!
//! use mlearn::model_builder::ModelBuilder;
//! use mlearn::linear_regression::{LinearMatrixSolver, ZeroInitializer};
//!
//! fn main() {
//!    let my_data = array![[1., 1.,], [1., 0.], [1., 2.]];
//!    let my_results = array![1., 1.1, 1.3];
//!    let builder = ModelBuilder {
//!        initializer: ZeroInitializer {},
//!        optimizer: LinearMatrixSolver {},
//!    };
//!    let model = builder.build_regression(&my_data, &my_results, None)
//!        .expect("Build failed.");
//!    println!("{}", model.coefficients);
//! }
//! ```

// Basic tools
pub mod metrics;

// Regression
pub mod regression;
pub mod linear_regression;

// Classification
pub mod classification;
pub mod logistic_regression;

// High level model interfaces
pub mod model_builder;
