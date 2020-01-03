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

#[cfg(test)]
mod tests {
    extern crate netlib_src;
    use ndarray::array;

    #[test]
    fn test_mean_squared_error() {
        let myarray1 = ndarray::array![1., 2., 3.];
        let myarray2 = ndarray::array![1., 3., 5.];
        let error = super::mean_squared_error(&myarray1, &myarray2);
        let epsilon = 1e-6;
        let expected_result: f64 = 5./3.;
        assert_eq!( (error - expected_result).abs() < epsilon, true);
    }

    #[test]
    fn test_mean_absolute_error() {
        let myarray1 = ndarray::array![1., 2., 3.];
        let myarray2 = ndarray::array![1., 3., 5.];
        let error = super::mean_absolute_error(&myarray1, &myarray2);
        let epsilon = 1e-6;
        let expected_result: f64 = 1.;
        assert_eq!( (error - expected_result).abs() < epsilon, true);
    }
}
