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
        inputs.dot(&self.coefficients)
    }
}

/// An initializer that simply initializes all coefficients to zero.
pub struct ZeroInitializer {}

impl Initialize for ZeroInitializer {
    type ModelType = LinearRegressor<f64>;

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
        S1: Data<Elem = f64>,
        S2: Data<Elem = f64>,
     {
        let n_features = inputs.shape()[1];
        let coefficients = Array::zeros((n_features,));
        LinearRegressor { coefficients }
    }
}


/// A solver that runs the basic matrix solution for a linear regression.
///
/// No regularization is done.
/// This is equivalent to running Newton's method on the gradient
/// of the MSE loss.
///
/// For a problem of the form `Ax = y`, this calculates:
/// `x = (A^t A)^{-1} A^t y`
pub struct LinearMatrixSolver {}

impl Optimize for LinearMatrixSolver {
    type ModelType = LinearRegressor<f64>;

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
        S1: Data<Elem = f64>,
        S2: Data<Elem = f64>,
    {
        let inputs_squared_inv = (inputs.t().dot(inputs)).inv()?;
        let inputs_outputs = inputs.t().dot(outputs);
        let results = inputs_squared_inv.dot(&inputs_outputs);
        model.coefficients = results;
        Ok(())
    }
}


/// An optimizer doing matrix-inversion based optimization
/// with L2-regularization.
///
/// Equivalent to a simple Newton's method solver.
pub struct RegularizedMatrixSolver {
    pub regularization_strength: f64,
    pub regularize_bias: bool,
}


impl Optimize for RegularizedMatrixSolver {
    type ModelType = LinearRegressor<f64>;

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
        S1: Data<Elem = f64>,
        S2: Data<Elem = f64>,
    {
        let count_norm = inputs.shape()[0] as f64;
        let mut regularization = self.regularization_strength * Array::eye(model.coefficients.len());
        if !self.regularize_bias {
            regularization[[0, 0]] = 0.;
        }
        let inputs_squared_inv = (
            inputs.t().dot(inputs) / count_norm + regularization
        ).inv()?;
        let inputs_outputs = inputs.t().dot(outputs) / count_norm;
        let results = inputs_squared_inv.dot(&inputs_outputs);
        model.coefficients = results;
        Ok(())
    }
}




#[cfg(test)]
mod tests {
    extern crate netlib_src;
    use ndarray::array;
    use super::*;

    #[test]
    fn test_linear_regressor() {
        let coefficients = array![1.0, 2.0, 3.0];
        let data = array![[0., 0., 0.], [0., 1., 2.]];
        let model = LinearRegressor::<f64> { coefficients };
        let predictions = model.predict(&data);
        let expected_results = array![0., 8.];
        let differences = expected_results - predictions;
        let epsilon: f64 = 1.0e-6;
        let good_results = differences.mapv(|a| a.abs() < epsilon);
        for &result in good_results.iter() {
            assert_eq!(result, true);
        }
    }

    #[test]
    fn test_zero_initializer() {
        let data = array![[0., 0., 0.], [0., 1., 2.]];
        let outputs = array![0., 0.];
        let initializer = ZeroInitializer{};
        let model = initializer.initialize(&data, &outputs, None);
        let expected_results = array![0., 0., 0.];
        assert_eq!(expected_results, model.coefficients);
    }

    #[test]
    fn test_linear_matrix_solver() {
        let data = array![[1., 0.], [1., 1.], [1., 2.]];
        let outputs = array![1., 3., 5.];
        let mut model = LinearRegressor {
            coefficients: array![0., 0.],
        };
        let solver = LinearMatrixSolver{};
        solver.optimize(&data, &outputs, None, &mut model)
            .expect("Failed to optimize.");
        let expected_coefficients = array![1.0, 2.0];
        let differences = expected_coefficients - model.coefficients;
        let absolute_error = differences.mapv(|a| a.abs()).sum();
        let epsilon = 1e-6;
        assert!(absolute_error < epsilon);
    }

    #[test]
    #[should_panic]
    fn test_solve_singular_matrix() {
        // Set up a singular matrix - perfectly colinear features
        let data = array![[1., 1.], [2., 2.], [3., 3.]];
        let outputs = array![1., 2., 3.];
        let mut model = LinearRegressor {
            coefficients: array![0., 0.],
        };
        let solver = LinearMatrixSolver{};
        // matrix inversion should fail and throw an error here
        solver.optimize(&data, &outputs, None, &mut model)
            .expect("Failed to optimize.");
        assert!(true);
    }

    #[test]
    fn test_regularized_matrix_solver() {
        let data = array![[1., 0.], [1., 1.]];
        let outputs = array![1., 3.];
        let mut model = LinearRegressor {
            coefficients: array![0., 0.],
        };
        let solver = RegularizedMatrixSolver{
            regularization_strength: 1.0,
            regularize_bias: false,
        };
        solver.optimize(&data, &outputs, None, &mut model)
            .expect("Failed to optimize.");
        let expected_coefficients = array![9. / 5., 2. / 5.];
        let differences = &expected_coefficients - &model.coefficients;
        println!("{}", model.coefficients);
        let absolute_error = differences.mapv(|a| a.abs()).sum();
        let epsilon = 1e-6;
        assert!(absolute_error < epsilon);
    }
}
