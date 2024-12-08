use crate::{
    utils::{
        geometry::Geometry,
        ode::{OdeProblem, OdeSolver, RungeKutta4},
    },
    write_datasets,
};
use std::f64::consts::PI;
pub mod aerodynamics;
pub mod bullet;

use aerodynamics::{Aerodynamics, Coefficients};
use bullet::Bullet;
use hdf5::File as Hdf5File;
use nalgebra::{Vector1, Vector3};
use pyo3::prelude::*;
use struct_iterable::Iterable;

#[pyclass(module = "stab_rs.stanag")]
#[derive(Clone, Iterable)]
struct SimData {
    // simulation states
    // position
    #[pyo3(get, set)]
    x1: Vec<f64>,
    #[pyo3(get, set)]
    x2: Vec<f64>,
    #[pyo3(get, set)]
    x3: Vec<f64>,
    // speed
    #[pyo3(get, set)]
    u1: Vec<f64>,
    #[pyo3(get, set)]
    u2: Vec<f64>,
    #[pyo3(get, set)]
    u3: Vec<f64>,
    #[pyo3(get, set)]
    udot1: Vec<f64>,
    #[pyo3(get, set)]
    udot2: Vec<f64>,
    #[pyo3(get, set)]
    udot3: Vec<f64>,
    #[pyo3(get, set)]
    roll: Vec<f64>,
    #[pyo3(get, set)]
    roll_rate: Vec<f64>,
    #[pyo3(get, set)]
    alpha_e_1: Vec<f64>,
    #[pyo3(get, set)]
    alpha_e_2: Vec<f64>,
    #[pyo3(get, set)]
    alpha_e_3: Vec<f64>,
    #[pyo3(get, set)]
    time: Vec<f64>,
}

#[pymethods]
impl SimData {
    #[new]
    fn new() -> Self {
        SimData {
            x1: Vec::new(),
            x2: Vec::new(),
            x3: Vec::new(),
            u1: Vec::new(),
            u2: Vec::new(),
            u3: Vec::new(),
            udot1: Vec::new(),
            udot2: Vec::new(),
            udot3: Vec::new(),
            roll: Vec::new(),
            roll_rate: Vec::new(),
            alpha_e_1: Vec::new(),
            alpha_e_2: Vec::new(),
            alpha_e_3: Vec::new(),
            time: Vec::new(),
        }
    }
}

#[pyclass(module = "stab_rs.stanag")]
pub struct Position {
    #[pyo3(get, set)]
    altitude: f64,
    #[pyo3(get, set)]
    latitude: f64,
    #[pyo3(get, set)]
    azimut: f64,
    #[pyo3(get, set)]
    quad_elev: f64,
    #[pyo3(get, set)]
    delta_azi: f64,
}

#[pymethods]
impl Position {
    #[new]
    fn new(altitude: f64, latitude: f64, azimut: f64, quad_elev: f64, delta_azi: f64) -> Self {
        Position {
            altitude,
            latitude,
            azimut,
            quad_elev,
            delta_azi,
        }
    }
}

impl Clone for Position {
    fn clone(&self) -> Self {
        Self {
            altitude: self.altitude.clone(),
            latitude: self.latitude.clone(),
            azimut: self.azimut.clone(),
            quad_elev: self.quad_elev.clone(),
            delta_azi: self.delta_azi.clone(),
        }
    }
}

#[pyclass(module = "stab_rs.stanag")]
struct Simulation {
    #[pyo3(get, set)]
    geometry: Geometry,
    #[pyo3(get, set)]
    position: Position,
    #[pyo3(get, set)]
    range: f64,
    #[pyo3(get, set)]
    twist_rate: f64,
    #[pyo3(get, set)]
    delta_yaw: f64,
    #[pyo3(get, set)]
    roll_rate: f64,
    #[pyo3(get, set)]
    init_vel: f64,
    #[pyo3(get, set)]
    iterations: usize,
    #[pyo3(get, set)]
    vec_data: SimData,
    #[pyo3(get, set)]
    coriolis: bool,
    bullet: Bullet,
}

#[pymethods]
impl Simulation {
    #[new]
    #[pyo3(signature = (geom, aero_path, position, range, vel, twist, delta_yaw, coriolis=None))]
    fn new(
        geom: Geometry,
        aero_path: String,
        position: Position,
        range: f64,
        vel: f64,
        twist: f64,
        delta_yaw: f64,
        coriolis: Option<bool>,
    ) -> Self {
        let cor = coriolis.unwrap_or(true);
        let bullet = Bullet::new(aero_path, geom.clone(), position.clone(), cor.clone());
        Simulation {
            geometry: geom,
            range: range,
            twist_rate: twist,
            delta_yaw: delta_yaw,
            roll_rate: 2.0 * PI / 0.0254 / twist * vel,
            init_vel: vel,
            position: position,
            iterations: 0,
            vec_data: SimData::new(),
            bullet: bullet,
            coriolis: cor,
        }
    }

    fn change_aerodata(&mut self, path: String) {
        self.bullet.coeffs = Coefficients::from_path(&path);
    }

    fn change_geometry(&mut self, geom: Geometry) {
        self.geometry = geom.clone();
        self.bullet.geom = geom;
    }

    fn run(&mut self) {
        let u0 = Vector3::new(
            self.position.quad_elev.cos() * self.position.delta_azi.cos(),
            self.position.quad_elev.sin(),
            self.position.quad_elev.cos() * self.position.delta_azi.sin(),
        ) * self.init_vel;

        self.bullet.state.vel_mut().set_column(0, &u0);
        self.bullet
            .state
            .roll_rate_mut()
            .set_column(0, &Vector1::new(self.roll_rate));

        let mut range = 0.0;
        let mut time = 0.0;
        while range <= self.range {
            let next = RungeKutta4.solve(&self.bullet, 0.0, 0.001, self.bullet.state.0);

            let aero = Aerodynamics::new(
                next.fixed_rows::<3>(3).clone_owned(),
                Vector3::zeros(),
                next.fixed_rows::<1>(7).clone_owned().x,
                self.bullet.alpha_e,
                self.geometry.diameter,
                self.geometry.area,
                next.fixed_rows::<1>(1).clone_owned().x,
            );

            // calculate derivatives
            let ode1 = self.bullet.odefun(0.0, next);

            // obtain and update alpha_e
            let alpha_e = aero.calc_alphae(&self.bullet.coeffs) * self.geometry.in_x;
            self.bullet.alpha_e = alpha_e.cross(&ode1.fixed_rows::<3>(3).clone_owned());
            
            self.bullet.state.0 = next;

            range = (next.get(0).unwrap().clone().powf(2.0)
                + next.get(2).unwrap().clone().powf(2.0))
            .sqrt();

            time += 0.001;

            // store data
            {
                self.vec_data
                    .x1
                    .append(&mut vec![next.get(0).unwrap().clone()]);
                self.vec_data
                    .x2
                    .append(&mut vec![next.get(1).unwrap().clone()]);
                self.vec_data
                    .x3
                    .append(&mut vec![next.get(2).unwrap().clone()]);
                self.vec_data
                    .u1
                    .append(&mut vec![next.get(3).unwrap().clone()]);
                self.vec_data
                    .u2
                    .append(&mut vec![next.get(4).unwrap().clone()]);
                self.vec_data
                    .u3
                    .append(&mut vec![next.get(5).unwrap().clone()]);
                self.vec_data
                    .roll
                    .append(&mut vec![next.get(6).unwrap().clone()]);
                self.vec_data
                    .roll_rate
                    .append(&mut vec![next.get(7).unwrap().clone()]);
                self.vec_data
                    .udot1
                    .append(&mut vec![ode1.get(3).unwrap().clone()]);
                self.vec_data
                    .udot2
                    .append(&mut vec![ode1.get(4).unwrap().clone()]);
                self.vec_data
                    .udot3
                    .append(&mut vec![ode1.get(5).unwrap().clone()]);
                self.vec_data
                    .alpha_e_1
                    .append(&mut vec![alpha_e.get(0).unwrap().clone()]);
                self.vec_data
                    .alpha_e_2
                    .append(&mut vec![alpha_e.get(1).unwrap().clone()]);
                self.vec_data
                    .alpha_e_3
                    .append(&mut vec![alpha_e.get(2).unwrap().clone()]);
                self.vec_data.time.append(&mut vec![time]);
            }
        }
    }

    fn write_results(&self) {
        let file = Hdf5File::create("sim_stanag.hdf5").unwrap();

        write_datasets!(file, self.vec_data)
    }
}

#[pymodule(module = "stab_rs.stanag")]
fn stanag(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Geometry>()?;
    m.add_class::<Position>()?;
    m.add_class::<Simulation>()?;
    m.add_class::<SimData>()?;
    Ok(())
}

pub fn init_stanag(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let stanag_child = PyModule::new_bound(_py, "stab_rs.stanag")?;
    stanag(_py, &stanag_child)?;

    m.add("stanag", &stanag_child)?;

    _py.import_bound("sys")?
        .getattr("modules")?
        .set_item("stab_rs.stanag", stanag_child)?;

    Ok(())
}
