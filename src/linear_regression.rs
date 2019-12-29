//! # Linear Regression Models
//!
//! `linear_regression` includes various structs for building and
//! training linear regression models.

extern crate ndarray;
extern crate ndarray_linalg;

use std::error::Error;

use ndarray::{Array, Array1, ArrayBase, Data, Ix1, Ix2, LinalgScalar};
use ndarray_linalg::solve::Inverse;

use crate::regression::{Initialize, Optimize, Regression};

/// Basic linear regression model.
///
/// Just contains a list of coefficients, with a bias included
/// as just a regular coefficient.
pub struct LinearRegressor<T: LinalgScalar> {
    pub coefficients: Array1<T>,
}

impl<T: LinalgScalar> Regression for LinearRegressor<T> {
    type DataType = T; // The output data type

    /// Predict the expected result for a series of samples.
    fn predict<S: Data<Elem = Self::DataType>>(
        &self,
        inputs: &ArrayBase<S, Ix2>,
    ) -> Array1<Self::DataType> {
        return inputs.dot(&self.coefficients)
    }
}

/// An initializer that simply initializes all coefficients to zero.
pub struct ZeroInitializer {}

impl Initialize for ZeroInitializer {
    type ModelType = LinearRegressor<f32>;

    /// Initialize a model.
    ///
    /// The number of features is extracted from the input data.
    fn initialize<S1, S2> (
        &self,
        inputs: &ArrayBase<S1, Ix2>,
        _outputs: &ArrayBase<S2, Ix1>,
        _weights: Option<&ArrayBase<S2, Ix1>>,
    ) -> Self::ModelType
    where
        S1: Data<Elem = f32>,
        S2: Data<Elem = f32>,
     {
        let n_features = inputs.shape()[1];
        let coefficients = Array::zeros((n_features,));
        let model = LinearRegressor {
            coefficients: coefficients,
        };
        model
    }
}


/// A solver that runs the basic matrix solution for a linear regression.
///
/// No regularization is done.
///
/// For a problem of the form `Ax = y`, this calculates:
/// `x = (A^t A)^{-1} A^t y`
pub struct LinearMatrixSolver {}

impl Optimize for LinearMatrixSolver {
    type ModelType = LinearRegressor<f32>;

    /// Train the model.
    ///
    /// Weights not used here.
    fn optimize<S1, S2>(
        &self,
        inputs: &ArrayBase<S1, Ix2>,
        outputs: &ArrayBase<S2, Ix1>,
        _weights: Option<&ArrayBase<S2, Ix1>>,
        model: &mut Self::ModelType,
    ) -> Result<(), Box<dyn Error>>
    where
        S1: Data<Elem = f32>,
        S2: Data<Elem = f32>,
    {
        let inputs_squared_inv = (inputs.t().dot(inputs)).inv()?;
        let inputs_outputs = inputs.t().dot(outputs);
        let results = inputs_squared_inv.dot(&inputs_outputs);
        model.coefficients = results;
        Ok(())
    }
}
