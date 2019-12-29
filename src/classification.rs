extern crate ndarray;

use ndarray::{Array1, Array2, ArrayBase, Data, LinalgScalar, Ix1, Ix2};

use std::error::Error;

pub type ClassLabel = usize;

pub trait Classification {
    type DataType: LinalgScalar;

    fn predict<S>(&self, inputs: &ArrayBase<S, Ix2>) -> Array1<ClassLabel>
    where
        S: Data<Elem = Self::DataType>;
}

pub trait ClassProbability: Classification {
    fn predict_proba<S>(&self, inputs: &ArrayBase<S, Ix2>) -> Array2<Self::DataType>
    where
        S: Data<Elem = Self::DataType>;
}

pub trait Initialize {
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

pub trait Optimize {
    type ModelType: Classification;

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
