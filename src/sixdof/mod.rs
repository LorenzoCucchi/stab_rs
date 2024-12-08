pub mod aerodynamics;
pub mod atmosphere;
pub mod bullet;
use aerodynamics::{AeroState, Aerodynamics, Coefficients};
use atmosphere::AtmosphereIsa;
use nalgebra::{Quaternion, Unit, UnitQuaternion, Vector3};
use pyo3::prelude::*;
use std::f64::consts::PI;
use struct_iterable::Iterable;

use crate::{
    utils::{
        geometry::Geometry,
        ode::{OdeProblem, OdeSolver, RungeKutta4},
    },
    write_datasets,
};

use bullet::Bullet;
use hdf5::File as Hdf5File;

#[pyclass(module = "stab_rs.sixdof")]
#[derive(Clone, Iterable)]
struct SimData {
    // NED Position
    #[pyo3(get, set)]
    x_n: Vec<f64>,
    #[pyo3(get, set)]
    x_e: Vec<f64>,
    #[pyo3(get, set)]
    x_d: Vec<f64>,
    // NED Velocity
    #[pyo3(get, set)]
    v_n: Vec<f64>,
    #[pyo3(get, set)]
    v_e: Vec<f64>,
    #[pyo3(get, set)]
    v_d: Vec<f64>,
    // BODY Velocity
    #[pyo3(get, set)]
    v_bx: Vec<f64>,
    #[pyo3(get, set)]
    v_by: Vec<f64>,
    #[pyo3(get, set)]
    v_bz: Vec<f64>,
    // QUATERNION
    #[pyo3(get, set)]
    quat_x: Vec<f64>,
    #[pyo3(get, set)]
    quat_y: Vec<f64>,
    #[pyo3(get, set)]
    quat_z: Vec<f64>,
    #[pyo3(get, set)]
    quat_w: Vec<f64>,
    // EULER ANGLES
    #[pyo3(get, set)]
    phi: Vec<f64>,
    #[pyo3(get, set)]
    theta: Vec<f64>,
    #[pyo3(get, set)]
    psi: Vec<f64>,
    // ANGULAR VEL.
    #[pyo3(get, set)]
    p: Vec<f64>,
    #[pyo3(get, set)]
    q: Vec<f64>,
    #[pyo3(get, set)]
    r: Vec<f64>,
    // AERO ANGLES
    #[pyo3(get, set)]
    alpha: Vec<f64>,
    #[pyo3(get, set)]
    beta: Vec<f64>,
    #[pyo3(get, set)]
    time: Vec<f64>,
}

#[pymethods]
impl SimData {
    #[new]
    fn new() -> Self {
        SimData {
            x_n: Vec::new(),
            x_e: Vec::new(),
            x_d: Vec::new(),
            v_n: Vec::new(),
            v_e: Vec::new(),
            v_d: Vec::new(),
            v_bx: Vec::new(),
            v_by: Vec::new(),
            v_bz: Vec::new(),
            quat_x: Vec::new(),
            quat_y: Vec::new(),
            quat_z: Vec::new(),
            quat_w: Vec::new(),
            phi: Vec::new(),
            theta: Vec::new(),
            psi: Vec::new(),
            p: Vec::new(),
            q: Vec::new(),
            r: Vec::new(),
            alpha: Vec::new(),
            beta: Vec::new(),
            time: Vec::new(),
        }
    }
}

#[pyclass(module = "stab_rs.sixdof")]
struct Simulation {
    #[pyo3(get, set)]
    dt: f64,
    #[pyo3(get, set)]
    ft: f64,
    is_init: bool,
    #[pyo3(get, set)]
    geometry: Geometry,
    #[pyo3(get, set)]
    twist_rate: f64,
    #[pyo3(get, set)]
    vec_data: SimData,
    aeropath: String,
    bullet: Bullet,
}

#[pymethods]
impl Simulation {
    #[new]
    fn new(geom: Geometry, aero_path: String, dt: f64, ft: f64, twist: f64) -> Self {
        Simulation {
            dt: dt,
            ft: ft,
            is_init: false,
            twist_rate: twist,
            vec_data: SimData::new(),
            bullet: Bullet::new(aero_path.clone(), geom.clone()),
            aeropath: aero_path,
            geometry: geom,
        }
    }

    fn change_aerodata(&mut self, path: String) {
        self.bullet.aerodynamics.coefficients = Coefficients::from_path(&path);
    }

    fn change_geometry(&mut self, geom: Geometry) {
        self.geometry = geom.clone();
        self.bullet.geom = geom;
    }

    fn init_state(
        &mut self,
        vx: f64,
        vy: f64,
        vz: f64,
        pitch: f64,
        roll: f64,
        yaw: f64,
        q: f64,
        r: f64,
    ) {
        let q_nb = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
        let p = 2.0 * PI / 0.0254 / self.twist_rate * vx;
        self.bullet
            .state
            .vel_n_mut()
            .set_column(0, &q_nb.transform_vector(&Vector3::new(vx, vy, vz)));
        self.bullet
            .state
            .quat_nb_vec_mut()
            .set_column(0, q_nb.as_vector());
        self.bullet
            .state
            .angvel_b_mut()
            .set_column(0, &Vector3::new(p, q, r));

        self.is_init = true;
    }

    fn run(&mut self) {
        if self.is_init {
            let mut time = 0.0;

            let coeffs = Coefficients::from_path(&self.aeropath);
            let atmosphere = Box::new(AtmosphereIsa::default());
            let aero = Aerodynamics::new(
                self.geometry.diameter,
                self.geometry.area,
                atmosphere,
                coeffs,
            );

            while time <= self.ft {
                let next = RungeKutta4.solve(&self.bullet, 0.0, self.dt, self.bullet.state.0);

                let q_nb: Unit<Quaternion<f64>> = UnitQuaternion::from_quaternion(
                    Quaternion::from_vector(next.fixed_rows::<4>(6).clone_owned()),
                );

                let vel_b: Vector3<f64> =
                    q_nb.inverse_transform_vector(&next.fixed_rows::<3>(3).clone_owned());
                let w_b: Vector3<f64> = next.fixed_rows::<3>(10).clone_owned();

                let aero_result = aero.calc(&AeroState::new(
                    vel_b,
                    Vector3::zeros(),
                    w_b.clone_owned(),
                    -next.fixed_rows::<3>(0)[2],
                ));

                let aero_angle = q_nb.inverse_transform_vector(
                    &Vector3::new(0.0, aero_result.alpha, aero_result.beta).clone_owned(),
                );

                self.bullet.state.0 = next;
                self.bullet.state.normalize_quat();

                time += self.dt;

                {
                    let pos_n = self.bullet.state.pos_n();
                    self.vec_data.x_n.append(&mut vec![pos_n.x]);
                    self.vec_data.x_e.append(&mut vec![pos_n.y]);
                    self.vec_data.x_d.append(&mut vec![pos_n.z]);

                    let vel_n = self.bullet.state.vel_n();
                    self.vec_data.v_n.append(&mut vec![vel_n.x]);
                    self.vec_data.v_e.append(&mut vec![vel_n.y]);
                    self.vec_data.v_d.append(&mut vec![vel_n.z]);

                    let vel_b = self
                        .bullet
                        .state
                        .quat_nb()
                        .inverse_transform_vector(&vel_n.clone_owned());
                    self.vec_data.v_bx.append(&mut vec![vel_b.x]);
                    self.vec_data.v_by.append(&mut vec![vel_b.y]);
                    self.vec_data.v_bz.append(&mut vec![vel_b.z]);

                    let q_nb = self.bullet.state.quat_nb();
                    let q_nb_vec = q_nb.as_vector();
                    self.vec_data.quat_x.append(&mut vec![q_nb_vec.x]);
                    self.vec_data.quat_y.append(&mut vec![q_nb_vec.y]);
                    self.vec_data.quat_z.append(&mut vec![q_nb_vec.z]);
                    self.vec_data.quat_w.append(&mut vec![q_nb_vec.w]);

                    let (roll, pitch, yaw) = q_nb.euler_angles();
                    self.vec_data.phi.append(&mut vec![roll]);
                    self.vec_data.theta.append(&mut vec![pitch]);
                    self.vec_data.psi.append(&mut vec![yaw]);

                    let ang_vel = self.bullet.state.angvel_b();
                    self.vec_data.p.append(&mut vec![ang_vel.x]);
                    self.vec_data.q.append(&mut vec![ang_vel.y]);
                    self.vec_data.r.append(&mut vec![ang_vel.z]);

                    self.vec_data.alpha.append(&mut vec![aero_angle.y]);
                    self.vec_data.beta.append(&mut vec![aero_angle.z]);

                    self.vec_data.time.append(&mut vec![time]);
                }
            }
        } else {
            panic!("Error: The simulation state is not initialized, use fn init_state");
        }
    }

    fn write_results(&self) {
        let file = Hdf5File::create("sim_sixdof.hdf5").unwrap();

        write_datasets!(file, self.vec_data)
    }
}

#[pymodule(module = "stab_rs.sixdof")]
fn sixdof(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Geometry>()?;
    m.add_class::<SimData>()?;
    m.add_class::<Simulation>()?;
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
