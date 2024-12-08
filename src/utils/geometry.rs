use nalgebra::Matrix3;
use pyo3::prelude::*;
use std::f64::consts::PI;

#[pyclass]
#[derive(Clone)]
pub struct Geometry {
    pub length: f64,
    pub diameter: f64,
    pub area: f64,
    pub in_x: f64,
    pub in_y: f64,
    pub xcg: f64,
    pub mass: f64,
    pub mass_grain: f64,
    pub inertia: Matrix3<f64>,
    pub inv_inertia: Matrix3<f64>,
}

#[pymethods]
impl Geometry {
    #[new]
    pub fn new(
        length: f64,
        diameter: f64,
        in_x: f64,
        in_y: f64,
        xcg: f64,
        mass_grain: f64,
    ) -> Self {
        let area = PI * diameter.powf(2.0) / 4.0;
        let mass = mass_grain / 15432.4;
        let inertia: Matrix3<f64> = Matrix3::new(in_x, 0.0, 0.0, 0.0, in_y, 0.0, 0.0, 0.0, in_y);
        Geometry {
            length,
            diameter,
            area,
            in_x,
            in_y,
            xcg,
            mass,
            mass_grain,
            inertia,
            inv_inertia: inertia.try_inverse().unwrap(),
        }
    }

    pub fn get_length(&self) -> PyResult<f64> {
        Ok(self.length)
    }

    pub fn set_length(&mut self, length: f64) -> PyResult<()> {
        self.length = length;
        Ok(())
    }

    fn get_diam(&self) -> PyResult<f64> {
        Ok(self.diameter)
    }

    fn set_diam(&mut self, diam: f64) -> PyResult<()> {
        self.diameter = diam;
        self.area = PI * self.diameter.powf(2.0) / 4.0;
        Ok(())
    }

    fn get_inx(&self) -> PyResult<f64> {
        Ok(self.in_x)
    }

    fn set_inx(&mut self, in_x: f64) -> PyResult<()> {
        self.in_x = in_x;
        Ok(())
    }

    fn get_iny(&self) -> PyResult<f64> {
        Ok(self.in_y)
    }

    fn set_iny(&mut self, in_y: f64) -> PyResult<()> {
        self.in_y = in_y;
        Ok(())
    }

    fn get_xcg(&self) -> PyResult<f64> {
        Ok(self.xcg)
    }

    fn set_xcg(&mut self, xcg: f64) -> PyResult<()> {
        self.xcg = xcg;
        Ok(())
    }

    fn get_mass(&self) -> PyResult<f64> {
        Ok(self.mass)
    }

    fn set_mass(&mut self, mass: f64) -> PyResult<()> {
        self.mass = mass;
        self.mass_grain = self.mass * 15432.4;
        Ok(())
    }

    fn get_mass_grain(&self) -> PyResult<f64> {
        Ok(self.mass)
    }

    fn set_mass_grain(&mut self, mass_grain: f64) -> PyResult<()> {
        self.mass_grain = mass_grain;
        self.mass = self.mass_grain / 15432.4;
        Ok(())
    }

    pub fn print(&self) {
        println!("Length: {} [m]", self.length);
        println!("Diameter: {} [m]", self.diameter);
        println!("Area: {} [m2]", self.area);
        println!("Inertia X: {}", self.in_x);
        println!("Inertia Y: {}", self.in_y);
        println!("XCG: {} [m]", self.xcg);
        println!("Mass: {} [Kg]", self.mass);
        println!("Mass Grain: {} [gr]", self.mass_grain);
    }
}

#[pymodule]
fn geometry(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Geometry>()?;
    Ok(())
}

pub fn init_geometry(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let linear_child = PyModule::new_bound(_py, "stab_rs.linear")?;
    geometry(_py, &linear_child)?;

    m.add("geometry", &linear_child)?;

    _py.import_bound("sys")?
        .getattr("modules")?
        .set_item("stab_rs.geometry", linear_child)?;

    Ok(())
}
