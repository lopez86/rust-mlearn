//! # Logistic regression
//!
//! `logistic_regression` provides tools to build and run
//! logistic regression models.
//!
//!

extern crate ndarray;

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2, NdFloat, s};

use crate::classification::{Classification, ClassLabel, ClassProbability};

pub struct LogisticRegression<T: NdFloat> {
    pub coefficients: Array1<T>,
}

fn convert_proba_to_class<T: NdFloat>(x: T, threshold: T) -> ClassLabel {
    if x > threshold {
        1
    } else {
        0
    }
}

impl<T: NdFloat> Classification for LogisticRegression<T> {
    type DataType = T;

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
