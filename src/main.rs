extern crate rusty_machine;
extern crate rulinalg;

use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::learning::SupModel;
use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;

fn main() {
    // Sample data
    let inputs = Matrix::new(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let targets = Vector::new(vec![2.0, 4.0, 6.0, 8.0]);

    // Create a linear regression model
    let mut lin_reg_model = LinRegressor::default();

    // Train the model
    lin_reg_model.train(&inputs, &targets).unwrap();

    // Predict new data
    let predictions = lin_reg_model.predict(&inputs).unwrap();

    println!("Predictions: {:?}", predictions);
}
