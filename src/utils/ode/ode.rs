use nalgebra::{RealField, SVector};

pub trait OdeSolver<T, const S: usize> {
    fn solve(
        &self,
        problem: &dyn OdeProblem<T, S>,
        t0: T,
        dt: T,
        y0: SVector<T, S>,
    ) -> SVector<T, S>;
}

pub trait OdeProblem<T, const S: usize>
where
    T: RealField,
{
    fn odefun(&self, t: T, y: SVector<T, S>) -> SVector<T, S>;
}

pub struct ForwardEuler;

impl<T: RealField, const S: usize> OdeSolver<T, S> for ForwardEuler {
    fn solve(
        &self,
        problem: &dyn OdeProblem<T, S>,
        t0: T,
        dt: T,
        y0: SVector<T, S>,
    ) -> SVector<T, S> {
        y0.clone() + problem.odefun(t0, y0) * dt
    }
}

pub struct RungeKutta4;

impl<T: RealField + From<f64> + Copy, const S: usize> OdeSolver<T, S> for RungeKutta4 {
    fn solve(
        &self,
        problem: &dyn OdeProblem<T, S>,
        t0: T,
        dt: T,
        y0: SVector<T, S>,
    ) -> SVector<T, S> {
        let hdt = dt / T::from(2.0);
        let k1 = problem.odefun(t0, y0);
        let k2 = problem.odefun(t0 + dt / 2.0.into(), y0 + k1 * hdt);
        let k3 = problem.odefun(t0 + hdt, y0 + k2 * hdt);
        let k4 = problem.odefun(t0 + dt, y0 + k3 * dt);

        y0 + (k1 + k2 * T::from(2.0) + k3 * T::from(2.0) + k4) * dt / T::from(6.0)
    }
}