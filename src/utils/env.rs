use std::default;
use std::f64::consts::E;
use std::fmt::DebugList;

use nalgebra::Vector3;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

#[pyfunction]
pub fn atmosisa(altitude: f64) -> (f64, f64, f64, f64) {
    let height = altitude;

    let g = 9.80665;
    let rs = 287.058;
    let r = 8.31447;
    let m = 0.0289644;

    let heights = [
        -610.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0,
    ];
    let temperatures = [
        292.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.87,
    ];
    let pressures = [
        108900.0, 22632.0, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.3734,
    ];

    let mut level = 0;
    for (i, &height_bound) in heights.iter().enumerate() {
        if height < height_bound {
            level = i - 1;
            break;
        }
    }

    let delta = height - heights[level];

    let lapse =
        (temperatures[level + 1] - temperatures[level]) / (heights[level + 1] - heights[level]);

    let temperature = temperatures[level] + delta * lapse;

    let pressure = if lapse != 0.0 {
        // Non-zero lapse rate
        pressures[level] * (1.0 + lapse * delta / temperatures[level]).powf(-g * m / (r * lapse))
    } else {
        // Zero lapse rate (isothermal layer)
        pressures[level] * (-g * m * delta / (r * temperature)).exp()
    };

    let density = pressure / (rs * temperature);

    let sound_speed = 331.3 * (1.0 + (temperature - 273.15) / 273.15).sqrt();

    (temperature, sound_speed, pressure, density)
}

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Clone)]
pub enum Turbulence {
    CONST = 0,
    MIL_1797 = 1,
    MIL_8785 = 2,
}

#[pyclass]
pub struct Wind {
    pub wind_vel: Vector3<f64>, //u, v, w
    pub u_old: f64,
    pub v_old: f64,
    pub w_old: f64,
    pub vel_body: f64,
    pub wind_20: f64,
    pub altitude: f64,
    pub delta_t: f64,
    pub length_w: f64,
    pub length_u: f64,
    pub length_v: f64,
    pub sigma_w: f64,
    pub sigma_u: f64,
    pub sigma_v: f64,
    pub turbulence: Turbulence,
    pub seed_w: [u8; 32],
    pub seed_u: [u8; 32],
    pub seed_v: [u8; 32],
    pub random_seed: bool,
}

#[pymethods]
impl Wind {
    #[new]
    pub fn new(
        wind_speed: f64,
        azimuth: f64,
        vert_wind: f64,
        altitude: f64,
        delta_t: f64,
        turb: Turbulence,
    ) -> Self {
        let n_wind = wind_speed * azimuth.cos();
        let e_wind = wind_speed * azimuth.sin();

        let wind_vel = Vector3::new(n_wind, e_wind, vert_wind);

        let mut seed_w: [u8; 32] = [0; 32];
        thread_rng().fill(&mut seed_w);
        let mut seed_u: [u8; 32] = [0; 32];
        thread_rng().fill(&mut seed_u);
        let mut seed_v: [u8; 32] = [0; 32];
        thread_rng().fill(&mut seed_v);

        Wind {
            wind_vel: wind_vel,
            u_old: 0.0,
            v_old: 0.0,
            w_old: 0.0,
            vel_body: 100.0,
            wind_20: wind_speed,
            altitude: altitude,
            delta_t: delta_t,
            length_w: 0.0,
            length_u: 0.0,
            length_v: 0.0,
            sigma_w: 0.0,
            sigma_u: 0.0,
            sigma_v: 0.0,
            turbulence: turb,
            seed_w: seed_w,
            seed_u: seed_u,
            seed_v: seed_v,
            random_seed: true,
        }
    }

    pub fn fixed_seed(&mut self, seed_u: [u8; 32], seed_v: [u8; 32], seed_w: [u8; 32]) {
        self.seed_u = seed_u;
        self.seed_v = seed_v;
        self.seed_w = seed_w;
        self.random_seed = false;
    }

    pub fn update_wind(&mut self, vel_body: f64, altitude: f64) {
        self.vel_body = vel_body;
        self.altitude = altitude;

        self.scale_lengths(altitude);
        self.turb_intensity(altitude);
        self.calc_wind();

        self.wind_vel.x += self.u_old;
        self.wind_vel.y += self.v_old;
        self.wind_vel.z += self.w_old;
    }

    fn scale_lengths(&mut self, altitude: f64) {
        let alt_ft = altitude * 3.28084;
        let mut length_u = 0.0;
        let mut length_v = 0.0;
        let mut length_w = 0.0;

        if self.turbulence != Turbulence::CONST {
            if alt_ft <= 1000.0 {
                length_u = alt_ft / (0.177 + 0.000823 * alt_ft).powf(1.2);
                if self.turbulence == Turbulence::MIL_1797 {
                    length_w = alt_ft / 2.0;
                    length_v = length_u / 2.0;
                } else if self.turbulence == Turbulence::MIL_8785 {
                    length_w = alt_ft;
                    length_v = length_u;
                }
            } else if alt_ft <= 2000.0 {
                length_u = 1000.0 / (0.177 + 0.000823 * 1000_f64).powf(1.2);
                length_w = 1000.0;
                length_v = length_u;
                let delta_x = 1000.0;
                if self.turbulence == Turbulence::MIL_1797 {
                    length_w = length_w / 2.0;
                    length_v = length_v / 2.0;
                    length_u = (1750.0 - length_u) / delta_x * (alt_ft - 1000.0) + length_u;
                    length_w = (875.0 - length_w) / delta_x * (alt_ft - 1000.0) + length_w;
                    length_v = (875.0 - length_v) / delta_x * (alt_ft - 1000.0) + length_v;
                } else if self.turbulence == Turbulence::MIL_8785 {
                    length_u = (1750.0 - length_u) / delta_x * (alt_ft - 1000.0) + length_u;
                    length_w = (1750.0 - length_w) / delta_x * (alt_ft - 1000.0) + length_w;
                    length_v = (1750.0 - length_v) / delta_x * (alt_ft - 1000.0) + length_v;
                }
            } else if alt_ft > 2000.0 {
                if self.turbulence == Turbulence::MIL_1797 {
                    length_u = 1750.0;
                    length_w = 875.0;
                    length_v = 875.0;
                } else if self.turbulence == Turbulence::MIL_8785 {
                    length_u = 1750.0;
                    length_w = 1750.0;
                    length_v = 1750.0;
                }
            }
        }

        self.length_u = length_u;
        self.length_w = length_w;
        self.length_v = length_v;
    }

    fn turb_intensity(&mut self, altitude: f64) {
        self.sigma_w = 0.1 * self.wind_20;
        self.sigma_u = 1.0 / (0.177 + 0.000823 * altitude * 3.28084).powf(0.4) * self.sigma_w;
        self.sigma_v = self.sigma_u;
        // TODO: implement turbulence intensity at medium high altitude
    }

    fn calc_wind(&mut self) {
        let vel = self.vel_body * 3.28084;

        let a1_u = -vel / self.length_u;
        let b1_u = self.sigma_u * (2.0 * vel / self.length_u).sqrt();

        let a1_v = -vel / self.length_v;
        let b1_v = self.sigma_v * (2.0 * vel / self.length_v).sqrt();

        let a1_w = -vel / self.length_w;
        let b1_w = self.sigma_w * (2.0 * vel / self.length_w).sqrt();

        let eta_u = Normal::new(0.0, 1.0)
            .unwrap()
            .sample(&mut StdRng::from_seed(self.seed_u));
        let eta_v = Normal::new(0.0, 1.0)
            .unwrap()
            .sample(&mut StdRng::from_seed(self.seed_v));
        let eta_w = Normal::new(0.0, 1.0)
            .unwrap()
            .sample(&mut StdRng::from_seed(self.seed_w));

        let rad_dt = self.delta_t.sqrt();

        self.u_old = self.u_old + a1_u * self.u_old * self.delta_t + b1_u * rad_dt * eta_u;
        self.v_old = self.v_old + a1_v * self.v_old * self.delta_t + b1_v * rad_dt * eta_v;
        self.w_old = self.w_old + a1_w * self.w_old * self.delta_t + b1_w * rad_dt * eta_w;
    }
}
