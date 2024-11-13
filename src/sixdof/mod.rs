pub mod aerodynamics;
pub mod bullet;

use pyo3::prelude::*;

#[pymodule(module = "stab_rs.sixdof")]
fn sixdof(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

pub fn init_sixdof(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sixdof_child = PyModule::new_bound(_py, "stab_rs.sixdof")?;
    sixdof(_py, &sixdof_child)?;

    m.add("sixdof", &sixdof_child)?;

    _py.import_bound("sys")?
        .getattr("modules")?
        .set_item("stab_rs.sixdof", sixdof_child)?;

    Ok(())
}
