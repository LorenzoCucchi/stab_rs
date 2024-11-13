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
}

#[pymethods]
impl Geometry {
    #[new]
    pub fn new(length: f64, diameter: f64, in_x: f64, in_y: f64, xcg: f64, mass_grain: f64) -> Self {
        let area = PI * diameter.powf(2.0) / 4.0;
        let mass = mass_grain / 15432.4;
        Geometry {
            length,
            diameter,
            area,
            in_x,
            in_y,
            xcg,
            mass,
            mass_grain,
        }
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