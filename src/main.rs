extern crate ndarray;
extern crate netlib_src;

use ndarray::array;

use mlearn::regression::RegressionBuilder;
use mlearn::linear_regression::{LinearMatrixSolver, ZeroInitializer};


fn main() {
    let my_data = array![[1., 1.,], [1., 0.], [1., 2.]];
    let my_results = array![1., 1., 1.];
    let builder = RegressionBuilder {
        initializer: ZeroInitializer {},
        optimizer: LinearMatrixSolver {},
    };
    let model = builder.build_model(&my_data, &my_results, None).expect("Build failed.");
    println!("{}", model.coefficients);
}
