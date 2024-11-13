use ndarray::{ArrayD, IxDyn};
use pyo3::prelude::*;
use struct_iterable::Iterable;

#[pyclass]
#[derive(Clone, Iterable, Debug)]
pub struct NamedVec {
    #[pyo3(get)]
    pub name: &'static str,
    #[pyo3(get, set)]
    pub data: Vec<f64>,
}

#[macro_export]
macro_rules! write_datasets {
    ($file: expr, $vec_data:expr) => {
        for (name, data) in $vec_data.iter() {
            let vec = data.downcast_ref::<Vec<f64>>().unwrap();
            $file
                .new_dataset::<f64>()
                .shape(vec.len())
                .create(name)
                .unwrap()
                .write(&vec)
                .unwrap();
        }
    };
}
