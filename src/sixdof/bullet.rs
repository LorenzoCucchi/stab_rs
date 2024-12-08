use nalgebra::{
    Quaternion, SVector, Unit, UnitQuaternion, Vector3, Vector4, VectorView,
    VectorViewMut, U1, U13, U3, U4,
};

use crate::utils::{geometry::Geometry, ode::OdeProblem};

use super::{
    aerodynamics::{AeroState, Aerodynamics, Coefficients},
    atmosphere::AtmosphereIsa,
};

#[derive(Debug, Default)]
pub struct State(pub SVector<f64, 13>);

impl State {
    pub fn init() -> Self {
        let state: SVector<f64, 13> = SVector::zeros();
        Self(state)
    }

    pub fn pos_n(&self) -> VectorView<'_, f64, U3, U1, U13> {
        self.0.fixed_rows::<3>(0)
    }

    pub fn vel_n(&self) -> VectorView<'_, f64, U3, U1, U13> {
        self.0.fixed_rows::<3>(3)
    }

    pub fn quat_nb_vec(&self) -> VectorView<'_, f64, U4, U1, U13> {
        self.0.fixed_rows::<4>(6)
    }

    pub fn angvel_b(&self) -> VectorView<'_, f64, U3, U1, U13> {
        self.0.fixed_rows::<3>(10)
    }

    pub fn pos_n_mut(&mut self) -> VectorViewMut<'_, f64, U3, U1, U13> {
        self.0.fixed_rows_mut::<3>(0)
    }
    pub fn vel_n_mut(&mut self) -> VectorViewMut<'_, f64, U3, U1, U13> {
        self.0.fixed_rows_mut::<3>(3)
    }

    pub fn quat_nb_vec_mut(&mut self) -> VectorViewMut<'_, f64, U4, U1, U13> {
        self.0.fixed_rows_mut::<4>(6)
    }

    pub fn angvel_b_mut(&mut self) -> VectorViewMut<'_, f64, U3, U1, U13> {
        self.0.fixed_rows_mut::<3>(10)
    }

    pub fn quat_nb(&self) -> UnitQuaternion<f64> {
        UnitQuaternion::from_quaternion(Quaternion::from_vector(self.quat_nb_vec().clone_owned()))
    }

    pub fn normalize_quat(&mut self) {
        let n = self.quat_nb_vec().normalize();
        self.quat_nb_vec_mut().set_column(0, &n);
    }
}


pub struct Bullet {
    pub aerodynamics: Aerodynamics,
    pub geom: Geometry,
    pub state: State,
}

impl Bullet {
    pub fn new(path: String, geom: Geometry) -> Self {
        let coeffs = Coefficients::from_path(&path);
        let atmosphere = Box::new(AtmosphereIsa::default());
        let aero = Aerodynamics::new(geom.diameter, geom.area, atmosphere, coeffs);

        Bullet {
            aerodynamics: aero,
            geom: geom,
            state: State::init(),
        }
    }
}

impl OdeProblem<f64, 13> for Bullet {
    fn odefun(&self, t: f64, y: SVector<f64, 13>) -> SVector<f64, 13> {
        let state = State(y);
        let mut dstate = State::default();

        let q_nb: Unit<Quaternion<f64>> = state.quat_nb();
        let vel_b: Vector3<f64> = q_nb.inverse_transform_vector(&state.vel_n().clone_owned());
        let w_b: Vector3<f64> = state.angvel_b().clone_owned();

        let aero = self.aerodynamics.calc(&AeroState::new(
            vel_b,
            Vector3::zeros(),
            w_b.clone_owned(),
            -state.pos_n()[2],
        ));

        let f_n: Vector3<f64> = q_nb.transform_vector(&aero.forces);
        let m_b: Vector3<f64> = aero.moments;

        let acc_n: Vector3<f64> = f_n / self.geom.mass + gravity(state.pos_n().clone_owned());

        let qw: Quaternion<f64> =
            Quaternion::from_vector(Vector4::new(w_b[0] / 2.0, w_b[1] / 2.0, w_b[2] / 2.0, 0.0));
        let q_dot = q_nb.into_inner() * qw;

        let w_dot: Vector3<f64> =
            self.geom.inv_inertia * (m_b + (self.geom.inertia * w_b).cross(&w_b));

        dstate
            .pos_n_mut()
            .set_column(0, &state.vel_n().clone_owned());
        dstate.vel_n_mut().set_column(0, &acc_n);
        dstate.quat_nb_vec_mut().set_column(0, q_dot.as_vector());
        dstate.angvel_b_mut().set_column(0, &w_dot);

        dstate.0
    }
}

pub fn gravity(pos: Vector3<f64>) -> Vector3<f64> {
    let radius = 6.356766 * 10e6;
    Vector3::new(0.0, 0.0, 1.0 - 2.0 * pos[2] / radius) * 9.80665
}
