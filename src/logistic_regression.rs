//! # Logistic regression
//!
//! `logistic_regression` provides tools to build and run
//! logistic regression models.

extern crate ndarray;

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2, NdFloat, s};

use crate::classification::{Classification, ClassLabel, ClassProbability};

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
}
