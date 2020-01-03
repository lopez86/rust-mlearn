extern crate ndarray;
extern crate netlib_src;

use ndarray::array;

use mlearn::model_builder::ModelBuilder;
use mlearn::linear_regression::{LinearMatrixSolver, ZeroInitializer, LinearRegressor};
use mlearn::logistic_regression::LogisticRegression;
use mlearn::regression::Regression;
use mlearn::classification::{Classification, ClassProbability};

fn main() {
    let my_data = array![[1., 1.,], [1., 0.], [1., 2.]];
    let my_results = array![1., 1., 1.];
    let builder = ModelBuilder {
        initializer: ZeroInitializer {},
        optimizer: LinearMatrixSolver {},
    };
    let model = builder.build_regression(&my_data, &my_results, None).expect("Build failed.");
    println!("{}", model.coefficients);
    let preds1 = model.predict(&my_data);
    println!("{}", preds1);
    let model2 = LinearRegressor::<f32>{
        coefficients: array![1., 2.],
    };
    println!("Linear regression 2");
    let _preds2 = model2.predict(&my_data);
    println!("Log regression");

    let logreg = LogisticRegression::<f32>{
        coefficients: array![1., 2.]
    };
    let pred_proba = logreg.predict_proba(&my_data);
    println!("{}", pred_proba);
    let preds = logreg.predict(&my_data);
    println!("{}", preds);
}
