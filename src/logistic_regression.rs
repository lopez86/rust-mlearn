//! # Logistic regression
//!
//! `logistic_regression` provides tools to build and run
//! logistic regression models.

extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::{
    Array,
    Array1,
    Array2,
    ArrayBase,
    Data,
    Ix1,
    Ix2,
    NdFloat,
    s,
};
use ndarray_linalg::solve::Inverse;


use crate::classification::{
    Classification,
    ClassLabel,
    ClassProbability,
    Optimize,
};

use std::error::Error;

/// Basic binary classification logistic regression model
///
/// A bias feature is assumed to be the first feature in
/// functions where a bias exists.
pub struct LogisticRegression<T: NdFloat> {
    pub coefficients: Array1<T>,
}

/// Simple function to classify based on threshold.
fn convert_proba_to_class<T: NdFloat>(x: T, threshold: T) -> ClassLabel {
    if x > threshold {
        1
    } else {
        0
    }
}

impl<T: NdFloat> Classification for LogisticRegression<T> {
    type DataType = T;

    /// Predict the clas label.
    fn predict<S>(&self, inputs: &ArrayBase<S, Ix2>) -> Array1<ClassLabel>
    where
        S: Data<Elem = Self::DataType>
    {
        let threshold = T::from(0.5).unwrap();
        let probas = self.predict_proba(inputs);
        let proba_slice = probas.slice(s![.., 1]);
        proba_slice.mapv(|a| convert_proba_to_class(a, threshold))
    }

}

impl<T: NdFloat> ClassProbability for LogisticRegression<T> {
    /// Predict the probabilities for each class.
    fn predict_proba<S>(&self, inputs: &ArrayBase<S, Ix2>) -> Array2<T>
    where
        S: Data<Elem = T>
    {
        let one = T::from(1.0).unwrap();
        let length = inputs.shape()[0];
        let neglogits = -inputs.dot(&self.coefficients);
        let probabilities = neglogits.mapv(|a| one / (one + a.exp()));
        let mut results = Array2::zeros((length, 2));
        results.slice_mut(s![.., 1]).assign(&probabilities);
        let inverse_probabilities = probabilities.mapv(|a| one - a);
        results.slice_mut(s![.., 0]).assign(&inverse_probabilities);
        results
    }
}


/// A solver using a naive implementation of Newton's method.
///
/// Be very careful with this solver since results can be very unstable
/// if bad initial conditions are chosen.
/// This is likely an effect of the disappearing gradients problem
/// for the logistic function. This will tend to drive the bias toward
/// extremely large values.
///
/// Currently, L2 regularization can be used. L1 regularization will also
/// likely be implemented in the near future.
///
/// The model makes use of directly inverting the Hessian of the log loss
/// and so could be a big performance hit if large numbers of features are
/// used.
struct SimpleNewtonsMethodOptimizer {
    number_of_iterations: usize,
    l2_strength: f64,
    regularize_bias: bool,
}

impl Optimize for SimpleNewtonsMethodOptimizer {
    type ModelType = LogisticRegression<f64>;

    /// Optimize the model.
    fn optimize<S1, S2, S3>(
        &self,
        inputs: &ArrayBase<S1, Ix2>,
        outputs: &ArrayBase<S2, Ix1>,
        _weights: Option<&ArrayBase<S3, Ix1>>,
        model: &mut Self::ModelType,
    ) -> Result<(), Box<dyn Error>>
    where
    S1: Data<Elem = f64>,
    S2: Data<Elem = ClassLabel>,
    S3: Data<Elem = f64>
    {
        let float_outputs = outputs.mapv(|a| a as f64);
        let size = inputs.shape()[0];
        let size_scale = 1.0 / (inputs.shape()[0] as f64);
        let mut l2_mask = self.l2_strength * Array::ones((model.coefficients.shape()[0],));
        if !self.regularize_bias {
            l2_mask[0] = 0.0;
        }
        let l2_weight_hessian = Array::diag(&l2_mask);
        let ones: Array1::<f64> = Array::ones(size);
        // TODO: Clean up this code - probably some can be combined.
        for _iter in 1..self.number_of_iterations {
            let predictions = model.predict_proba(inputs);
            let predictions = predictions.slice(s![.., 1]);
            let diff = &float_outputs - &predictions;
            let gradients = - size_scale * inputs.t().dot(&diff) + &l2_mask * &model.coefficients;
            let inv_predictions = &ones - &predictions;
            let hessian_weight = &inv_predictions * &predictions;
            let hessian_weight = hessian_weight.into_shape((size, 1))?;
            let weighted_inputs = inputs * &hessian_weight;
            let transformed_inputs = inputs.t().dot(&weighted_inputs);
            // Note: Will be a bottleneck when large numbers of features are chosen
            let inv_hessian = (size_scale * &transformed_inputs + &l2_weight_hessian).inv()?;
            model.coefficients = &model.coefficients - &inv_hessian.dot(&gradients);
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    extern crate netlib_src;
    use super::*;
    use ndarray::array;

    #[test]
    fn test_logistic_regression_probabilities() {
        let model = LogisticRegression::<f64> {
            coefficients: array![1.0, 2.0],
        };
        let data = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.], [0., -1.]];
        let prob2 = 1. / (1. + (-2_f64).exp());
        let prob3 = 1. / (1. + (-1_f64).exp());
        let prob4 = 1. / (1. + (-3_f64).exp());
        let prob5 = 1. / (1. + (2_f64).exp());

        let expected_results = array![
            [0.5, 0.5],
            [1. - prob2, prob2],
            [1. - prob3, prob3],
            [1. - prob4, prob4],
            [1. - prob5, prob5],
        ];

        let result_probs = model.predict_proba(&data);
        let differences = expected_results - result_probs;
        let epsilon = 1e-8;
        let error = differences.mapv(|a| a.abs()).sum();
        assert!(error < epsilon);
    }

    #[test]
    fn test_logistic_regression_predictions() {
        let model = LogisticRegression::<f64> {
            coefficients: array![1.0, 2.0],
        };
        let data = array![
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.],
            [0., -1.],
            [-1., 0.],
            [-1., -1.]
        ];

        let expected_results = array![0, 1, 1, 1, 0, 0, 0];
        let results = model.predict(&data);
        assert_eq!(expected_results, results);
    }

    #[test]
    fn test_newtons_method_optimization() {
        let mut model = LogisticRegression::<f64> {
            coefficients: array![0.0, 0.0],
        };
        let data: Array2::<f64> = array![
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.25],
            [1.0, 0.25],
            [1.0, 0.25],
            [1.0, 0.25],
            [1.0, 0.25],
            [1.0, 0.5],
            [1.0, 0.5],
            [1.0, 0.5],
            [1.0, 0.5],
            [1.0, 0.5],
            [1.0, 0.75],
            [1.0, 0.75],
            [1.0, 0.75],
            [1.0, 0.75],
            [1.0, 0.75],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ];
        let outputs: Array1::<usize> = array![
            0, 0, 0, 0, 1,
            0, 0, 0, 1, 1,
            1, 1, 1, 0, 0,
            1, 1, 1, 1, 0,
            1, 1, 1, 1, 1,
        ];
        // Should figure out how to get the type to infer correctly
        let weights: Option<&Array1::<f64>> = None;
        let optimizer = SimpleNewtonsMethodOptimizer {
            number_of_iterations: 100,
            l2_strength: 0.02,
            regularize_bias: false,
        };
        optimizer.optimize(&data, &outputs, weights, &mut model)
            .expect("Could not optimize!");
        let expected_coefficients = array![-0.585770, 2.08941];
        let epsilon = 1e-5;
        let differences = expected_coefficients - model.coefficients;
        let error = differences.mapv(|a| a.abs()).sum();
        assert!(error < epsilon);
    }
}
