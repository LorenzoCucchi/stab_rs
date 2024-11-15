use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
use utils::geometry;

mod linear;
mod sixdof;
mod stanag;
mod utils;

#[pymodule(module = "stab_rs")]
fn stab_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    linear::init_linear(_py, m)?;
    stanag::init_stanag(_py, m)?;
    sixdof::init_sixdof(_py, m)?;
    geometry::init_geometry(_py, m)?;

    Ok(())
}
