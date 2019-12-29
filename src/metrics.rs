//! # Metrics
//!
//! `metrics` provides various evaluation metrics.
extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::{ArrayBase, Data, Ix1, NdFloat};

/// Calculate the mean squared error (L2 loss) between two arrays.
pub fn mean_squared_error<T, S1, S2>(
    truth: &ArrayBase<S1, Ix1>,
    predictions: &ArrayBase<S2, Ix1>
) -> T
where
    T: NdFloat,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let difference = truth - predictions;
    let squares = &difference * &difference;
    let mean = squares.sum() / T::from(squares.len()).unwrap();
    mean
}

/// Calculate the mean absolute error (L1 loss) between two arrays.
pub fn mean_absolute_error<T, S1, S2>(
    truth: &ArrayBase<S1, Ix1>,
    predictions: &ArrayBase<S2, Ix1>
) -> T
where
    T: NdFloat,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let difference = (truth - predictions).mapv_into(T::abs);
    let mean = difference.sum() / T::from(difference.len()).unwrap();
    mean
}
