use pyo3::prelude::*;

mod linear;
mod sixdof;
mod stanag;
mod utils;

#[pymodule]
fn stab_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    linear::init_linear(_py, m)?;
    stanag::init_stanag(_py, m)?;
    Ok(())
}
