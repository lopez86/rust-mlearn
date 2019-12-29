//! # Classification models
//!
//! `classification` provides basic types and traits to construct
//! models for classification.

extern crate ndarray;

use ndarray::{Array1, Array2, ArrayBase, Data, LinalgScalar, Ix1, Ix2};

use std::error::Error;

/// All labels should be the same type.
///
/// We want labels to be able to fit in arrays but we don't need
/// to be able to do math on them.
pub type ClassLabel = usize;


/// Minimal API for a classifiers.
///
/// It just needs to take some data and output labels.
pub trait Classification {
    /// The type of the input data, typically f32 or f64
    type DataType: LinalgScalar;

    /// Predict the most likely label for each input sample.
    ///
    /// Inputs are of the shape [n_samples, n_feauteres].
    /// Outputs are of the shape [n_samples].
    ///
    /// Ties must be resolved in the prediction function.
    fn predict<S>(&self, inputs: &ArrayBase<S, Ix2>) -> Array1<ClassLabel>
    where
        S: Data<Elem = Self::DataType>;
}

/// Trait to predict the per-class probabilities.
///
/// More detail than the basic Classification type but not always
/// possible for all classifiers.
pub trait ClassProbability: Classification {
    /// Predict the per-class probabilities.
    ///
    /// Inputs and outputs are of the shape [n_samples, n_features]
    fn predict_proba<S>(&self, inputs: &ArrayBase<S, Ix2>) -> Array2<Self::DataType>
    where
        S: Data<Elem = Self::DataType>;
}

/// Trait for initializing classification models.
pub trait Initialize {
    /// The type of model this initializer will act on.
    type ModelType: Classification;

    /// Initialize the model from some data.
    fn initialize<S1, S2, S3>(
        &self,
        inputs: &ArrayBase<S1, Ix2>,
        outputs: &ArrayBase<S2, Ix1>,
        weights: Option<&ArrayBase<S3, Ix1>>,
    ) -> Self::ModelType
    where
        S1: Data<Elem = <<Self as Initialize>::ModelType as Classification>::DataType>,
        S2: Data<Elem = ClassLabel>,
        S3: Data<Elem = <<Self as Initialize>::ModelType as Classification>::DataType>;
}

/// Optimize an already initialized classification model.
pub trait Optimize {
    /// The type of model this optimizer will act on.
    type ModelType: Classification;

    /// Optimize the model.
    fn optimize<S1, S2, S3>(
        &self,
        inputs: &ArrayBase<S1, Ix2>,
        outputs: &ArrayBase<S2, Ix1>,
        weights: Option<&ArrayBase<S3, Ix1>>,
        model: &mut Self::ModelType,
    ) -> Result<(), Box<dyn Error>>
    where
        S1: Data<Elem = <<Self as Optimize>::ModelType as Classification>::DataType>,
        S2: Data<Elem = ClassLabel>,
        S3: Data<Elem = <<Self as Optimize>::ModelType as Classification>::DataType>;
}
