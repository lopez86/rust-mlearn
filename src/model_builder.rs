//! # Model builder
//!
//! `model_builder` defines a basic class to create models from
//! an initializer and an optimizer.

extern crate ndarray;

use ndarray::{ArrayBase, Data, Ix1, Ix2};

use crate::regression::{Initialize as RegInit, Optimize as RegOpt, Regression};
use crate::classification::{
    Classification,
    ClassLabel,
    Initialize as ClassInit,
    Optimize as ClassOpt
};

use std::error::Error;

/// Basic struct to build models.
///
/// Takes an initializer and optimizer for a model and builds it.
/// A factory type structure is preferred to direct user creation
/// as it avoids the problem of having potentially untrained models.
pub struct ModelBuilder<I, T>
{
    pub initializer: I,
    pub optimizer: T,
}

/// Implementation for a factory class to build regression models.
impl<M, I, T> ModelBuilder<I, T> where
    M: Regression,
    I: RegInit<ModelType = M>,
    T: RegOpt<ModelType= M>,
{
    /// Builds a regression model.
    pub fn build_regression<
        S1: Data<Elem = M::DataType>,
        S2: Data<Elem = M::DataType>,
    >(
        &self,
        inputs: &ArrayBase<S1, Ix2>,
        outputs: &ArrayBase<S2, Ix1>,
        weights: Option<&ArrayBase<S2, Ix1>>,
    ) -> Result<M, Box<dyn Error>>
    {
        let mut model = self.initializer.initialize(inputs, outputs, weights);
        self.optimizer.optimize(inputs, outputs, weights, &mut model)?;
        Ok(model)
    }
}

/// Implementation for a factory class to build classifiers.
impl<M, I, T> ModelBuilder<I, T> where
    M: Classification,
    I: ClassInit<ModelType = M>,
    T: ClassOpt<ModelType= M>,
{
    /// Builds a classification model.
    pub fn build_classification<
        S1: Data<Elem = M::DataType>,
        S2: Data<Elem = ClassLabel>,
        S3: Data<Elem = M::DataType>,
    >(
        &self,
        inputs: &ArrayBase<S1, Ix2>,
        outputs: &ArrayBase<S2, Ix1>,
        weights: Option<&ArrayBase<S3, Ix1>>,
    ) -> Result<M, Box<dyn Error>>
    {
        let mut model = self.initializer.initialize(inputs, outputs, weights);
        self.optimizer.optimize(inputs, outputs, weights, &mut model)?;
        Ok(model)
    }
}
