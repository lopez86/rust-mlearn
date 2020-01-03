extern crate ndarray;
extern crate netlib_src;

use ndarray::array;

use mlearn::logistic_regression::LogisticRegression;
use mlearn::classification::{Classification, ClassProbability};

fn main() {
    let my_data = array![[1., 1.,], [1., 0.], [1., 2.]];
    let logreg = LogisticRegression::<f32>{
        coefficients: array![1., 2.]
    };
    let pred_proba = logreg.predict_proba(&my_data);
    println!("{}", pred_proba);
    let preds = logreg.predict(&my_data);
    println!("{}", preds);
}
