use nalgebra::{SVector, Vector1, Vector3, VectorView, VectorViewMut, U1, U3, U8};

use crate::utils::geometry::Geometry;
use crate::utils::ode::OdeProblem;

use super::{
    aerodynamics::{Aerodynamics, Coefficients},
    Position,
};

use pyo3::prelude::*;

#[derive(Debug, Default)]
pub struct State(pub SVector<f64, 8>);

impl State {
    pub fn init() -> Self {
        let state: SVector<f64, 8> = SVector::zeros();
        Self(state)
    }
    pub fn pos(&self) -> VectorView<'_, f64, U3, U1, U8> {
        self.0.fixed_rows::<3>(0)
    }
    pub fn vel(&self) -> VectorView<'_, f64, U3, U1, U8> {
        self.0.fixed_rows::<3>(3)
    }
    pub fn roll(&self) -> VectorView<'_, f64, U1, U1, U8> {
        self.0.fixed_rows::<1>(6)
    }
    pub fn roll_rate(&self) -> VectorView<'_, f64, U1, U1, U8> {
        self.0.fixed_rows::<1>(7)
    }
    pub fn pos_mut(&mut self) -> VectorViewMut<'_, f64, U3, U1, U8> {
        self.0.fixed_rows_mut::<3>(0)
    }
    pub fn vel_mut(&mut self) -> VectorViewMut<'_, f64, U3, U1, U8> {
        self.0.fixed_rows_mut::<3>(3)
    }
    pub fn roll_mut(&mut self) -> VectorViewMut<'_, f64, U1, U1, U8> {
        self.0.fixed_rows_mut::<1>(6)
    }
    pub fn roll_rate_mut(&mut self) -> VectorViewMut<'_, f64, U1, U1, U8> {
        self.0.fixed_rows_mut::<1>(7)
    }
}

#[pyclass]
pub struct Bullet {
    pub coeffs: Coefficients,
    pub position: Position,
    pub geom: Geometry,
    pub state: State,
    pub alpha_e: Vector3<f64>,
}

#[pymethods]
impl Bullet {
    #[new]
    pub fn new(path: String, geom: Geometry, pos: Position) -> Self {
        let coeffs = Coefficients::from_path(&path);

        Bullet {
            state: State::init(),
            position: pos,
            coeffs: coeffs,
            geom: geom,
            alpha_e: Vector3::zeros(),
        }
    }
}

impl OdeProblem<f64, 8> for Bullet {
    fn odefun(&self, t: f64, y: SVector<f64, 8>) -> SVector<f64, 8> {
        let state = State(y);
        let mut dstate = State::default();

        let vel = state.vel().clone_owned();
        let pos = state.pos().clone_owned();

        let aero = Aerodynamics::new(
            vel,
            Vector3::zeros(),
            state.roll_rate().clone_owned().x,
            self.alpha_e.clone_owned(),
            self.geom.diameter.clone(),
            self.geom.area.clone(),
        );

        let (force, force_roll) = aero.actions(&self.coeffs);
        let cor_acc = OMEGA
            * Vector3::new(
                self.position.latitude.cos() * self.position.azimut.cos(),
                self.position.latitude.cos(),
                -self.position.latitude.cos() * self.position.azimut.sin(),
            );

        let acc = force / self.geom.mass + gravity(pos) + cor_acc;
        let pdot = Vector1::new(force_roll / self.geom.in_x);

        dstate.pos_mut().set_column(0, &state.vel().clone_owned());
        dstate.vel_mut().set_column(0, &acc);
        dstate
            .roll_mut()
            .set_column(0, &state.roll_rate().clone_owned());
        dstate.roll_rate_mut().set_column(0, &pdot);

        dstate.0
    }
}

pub fn gravity(pos: Vector3<f64>) -> Vector3<f64> {
    let R = 6.356766 * 10e6;
    -Vector3::new(pos[0] / R, 1.0 - 2.0 * pos[1] / R, pos[2] / R) * 9.80665
}

const OMEGA: f64 = 7.292115 * 10e-5;
