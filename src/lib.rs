use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods}};

mod linear;
mod sixdof;
mod stanag;
mod utils;


#[pymodule(module = "stab_rs")]
fn stab_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    linear::init_linear(_py, m)?;
    stanag::init_stanag(_py, m)?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);