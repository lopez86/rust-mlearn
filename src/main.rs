extern crate ndarray;
extern crate netlib_src;

use ndarray::array;

use mlearn::model_builder::ModelBuilder;
use mlearn::linear_regression::{LinearMatrixSolver, ZeroInitializer, LinearRegressor};
use mlearn::metrics::{mean_absolute_error, mean_squared_error};
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
    let preds2 = model2.predict(&my_data);
    println!("Log regression");

    // TODO: Turn into unit tests
    let myarray1 = array![1., 2., 3.];
    let myarray2 = array![1., 3., 5.];

    let logreg = LogisticRegression::<f32>{
        coefficients: array![1., 2.]
    };
    let pred_proba = logreg.predict_proba(&my_data);
    println!("{}", pred_proba);
    let preds = logreg.predict(&my_data);
    println!("{}", preds);

    let l1_loss = mean_absolute_error(&myarray1, &myarray2);
    let l2_loss = mean_squared_error(&myarray1, &myarray2);
    println!("{}, {}", l1_loss, l2_loss);
    println!("Should be: 1, 5/3");
}
