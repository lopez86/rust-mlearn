extern crate ndarray;
extern crate netlib_src;

use ndarray::array;

use mlearn::regression::RegressionBuilder;
use mlearn::linear_regression::{LinearMatrixSolver, ZeroInitializer};
use mlearn::metrics::{mean_absolute_error, mean_squared_error};


fn main() {
    let my_data = array![[1., 1.,], [1., 0.], [1., 2.]];
    let my_results = array![1., 1., 1.];
    let builder = RegressionBuilder {
        initializer: ZeroInitializer {},
        optimizer: LinearMatrixSolver {},
    };
    let model = builder.build_model(&my_data, &my_results, None).expect("Build failed.");
    println!("{}", model.coefficients);

    // TODO: Turn into unit tests
    let myarray1 = array![1., 2., 3.];
    let myarray2 = array![1., 3., 5.];

    let l1_loss = mean_absolute_error(&myarray1, &myarray2);
    let l2_loss = mean_squared_error(&myarray1, &myarray2);
    println!("{}, {}", l1_loss, l2_loss);
    println!("Should be: 1, 5/3");
}
