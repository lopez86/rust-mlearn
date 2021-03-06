//! # Regression models
//!
//! `regression` provides traits to build and run regression models.
//!
extern crate ndarray;

use std::error::Error;

use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2, LinalgScalar};

/// Basic API for a regression model using ndarray types.
///
/// The model here takes in a 2-dimensional array and returns
/// a 1-dimensional array of predictions.
pub trait Regression {
    type DataType: LinalgScalar;

    /// Predict the results of the inputs.
    ///
    /// Inputs are assumed to be in the shape [n_samples, n_features],
    /// and outputs are in the shape [n_samples]. Inputs and outputs have
    /// the same basic data type, typically f32 or f64.
    fn predict<S>(&self, inputs: &ArrayBase<S, Ix2>) -> Array1<Self::DataType>
    where
        S: Data<Elem = Self::DataType>;
}

/// A trait to define a model training interface.
///
/// Training interfaces are assumed to be separate from the Regression
/// object: A training object takes a Regression and modifies it based
/// on the data given.
pub trait Optimize {
    type ModelType: Regression;

    fn optimize<S1, S2>(
        &self,
        inputs: &ArrayBase<S1, Ix2>,
        outputs: &ArrayBase<S2, Ix1>,
        weights: Option<&ArrayBase<S2, Ix1>>,
        model: &mut Self::ModelType,
    ) -> Result<(), Box<dyn Error>>
    where
        S1: Data<Elem = <<Self as Optimize>::ModelType as Regression>::DataType>,
        S2: Data<Elem = <<Self as Optimize>::ModelType as Regression>::DataType>;
}

/// A trait to initialize a model.
pub trait Initialize {
    type ModelType: Regression;

    /// Initialize the model from some data.
    fn initialize<S1, S2>(
        &self,
        inputs: &ArrayBase<S1, Ix2>,
        outputs: &ArrayBase<S2, Ix1>,
        weights: Option<&ArrayBase<S2, Ix1>>,
    ) -> Self::ModelType
    where
        S1: Data<Elem = <<Self as Initialize>::ModelType as Regression>::DataType>,
        S2: Data<Elem = <<Self as Initialize>::ModelType as Regression>::DataType>;
}
