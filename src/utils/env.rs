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
#[derive(Clone)]
pub struct Wind {
    pub wind_vel: Vector3<f64>, //u, v, w
    pub wind_vel_old: Vector3<f64>,
    wind_vel_0: Vector3<f64>,
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
    method: usize,
    pub rng_u: StdRng,
    pub rng_v: StdRng,
    pub rng_w: StdRng,
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
        let wind_vel: Vector3<f64> = Vector3::new(wind_speed, vert_wind, azimuth);

        let mut seed_w: [u8; 32] = [0; 32];
        thread_rng().fill(&mut seed_w);
        let mut seed_u: [u8; 32] = [0; 32];
        thread_rng().fill(&mut seed_u);
        let mut seed_v: [u8; 32] = [0; 32];
        thread_rng().fill(&mut seed_v);

        Wind {
            wind_vel: wind_vel,
            wind_vel_old: Vector3::zeros(),
            wind_vel_0: wind_vel,
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
            method: 0,
            rng_u: StdRng::from_seed(seed_u),
            rng_v: StdRng::from_seed(seed_v),
            rng_w: StdRng::from_seed(seed_w),
        }
    }

    pub fn init_stanag(&mut self) {
        let wind_vel = self.wind_vel.x;
        let vert_win = self.wind_vel.y;
        let azimuth = self.wind_vel.z;

        let w_n = wind_vel * azimuth.cos();
        let w_e = wind_vel * azimuth.sin();

        self.wind_vel.x = w_n;
        self.wind_vel.y = vert_win;
        self.wind_vel.z = w_e;

        self.wind_vel_0 = self.wind_vel;
        self.method = 0;
    }

    pub fn init_sixdof(&mut self) {
        let wind_vel = self.wind_vel.x;
        let vert_win = self.wind_vel.y;
        let azimuth = self.wind_vel.z;

        let w_n = wind_vel * azimuth.cos();
        let w_e = wind_vel * azimuth.sin();

        self.wind_vel.x = w_n;
        self.wind_vel.y = w_e;
        self.wind_vel.z = -vert_win;
        println!("wind: {:?}", self.wind_vel);
        self.wind_vel_0 = self.wind_vel;
        self.method = 1;
    }

    pub fn fixed_seed(&mut self, seed_u: [u8; 32], seed_v: [u8; 32], seed_w: [u8; 32]) {
        self.seed_u = seed_u;
        self.seed_v = seed_v;
        self.seed_w = seed_w;
        self.random_seed = false;
        self.rng_u = StdRng::from_seed(seed_u);
        self.rng_v = StdRng::from_seed(seed_v);
        self.rng_w = StdRng::from_seed(seed_w);
    }

    pub fn update_wind(&mut self, vel_body: f64, altitude: f64) {
        self.vel_body = vel_body;
        self.altitude = altitude;
        self.wind_vel_old = self.wind_vel;

        self.scale_lengths(altitude);
        self.turb_intensity(altitude);

        self.calc_wind();

        if self.method == 0 {
            self.wind_vel.x = self.wind_vel_0.x + self.u_old;
            self.wind_vel.y = self.wind_vel_0.y + self.w_old;
            self.wind_vel.z = self.wind_vel_0.z + self.v_old;
        } else if self.method == 1 {
            self.wind_vel.x = self.wind_vel_0.x + self.u_old;
            self.wind_vel.y = self.wind_vel_0.y + self.v_old;
            self.wind_vel.z = self.wind_vel_0.z + self.w_old;
        }
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
        if self.turbulence == Turbulence::CONST {
            self.u_old = 0.0;
            self.v_old = 0.0;
            self.w_old = 0.0;
        } else {
            let vel = self.vel_body * 3.28084;

            let a1_u = -vel / self.length_u;
            let b1_u = self.sigma_u * (2.0 * vel / self.length_u).sqrt();

            let a1_v = -vel / self.length_v;
            let b1_v = self.sigma_v * (2.0 * vel / self.length_v).sqrt();

            let a1_w = -vel / self.length_w;
            let b1_w = self.sigma_w * (2.0 * vel / self.length_w).sqrt();

            let eta_u = Normal::new(0.0, 1.0).unwrap().sample(&mut self.rng_u);
            let eta_v = Normal::new(0.0, 1.0).unwrap().sample(&mut self.rng_v);
            let eta_w = Normal::new(0.0, 1.0).unwrap().sample(&mut self.rng_w);

            let rad_dt = self.delta_t.sqrt();

            self.u_old = self.u_old + a1_u * self.u_old * self.delta_t + b1_u * rad_dt * eta_u;
            self.v_old = self.v_old + a1_v * self.v_old * self.delta_t + b1_v * rad_dt * eta_v;
            self.w_old = self.w_old + a1_w * self.w_old * self.delta_t + b1_w * rad_dt * eta_w;
        }
    }
}

#[pymodule]
fn wind(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Wind>()?;
    m.add_class::<Turbulence>()?;
    Ok(())
}

pub fn init_wind(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let linear_child = PyModule::new_bound(_py, "stab_rs.stanag")?;
    wind(_py, &linear_child)?;

    m.add("wind", &linear_child)?;

    _py.import_bound("sys")?
        .getattr("modules")?
        .set_item("stab_rs.wind", linear_child)?;

    Ok(())
}

pub trait Atmosphere {
    fn pressure(&self, h: f64) -> f64;
    fn density(&self, h: f64) -> f64;
    fn temperature(&self, h: f64) -> f64;
    fn sound_speed(&self, h: f64) -> f64;
}

pub struct AtmosphereIsa {
    pressure_0: f64,
    temperature_0: f64,
    density_0: f64,
    h_0: f64,
    g_0: f64,
    specific_gas_constant: f64,
    a: f64,
}

impl Default for AtmosphereIsa {
    fn default() -> Self {
        AtmosphereIsa {
            pressure_0: 101325.0,
            temperature_0: 288.15,
            density_0: 1.2250,
            h_0: 0.0,
            g_0: 9.80665,
            specific_gas_constant: 287.052874,
            a: -0.0065,
        }
    }
}

impl AtmosphereIsa {
    fn new(
        pressure_0: f64,
        temperature_0: f64,
        density_0: f64,
        h_0: f64,
        g_0: f64,
        molar_gas_constant: f64,
        a: f64,
    ) -> AtmosphereIsa {
        AtmosphereIsa {
            pressure_0,
            temperature_0,
            density_0,
            h_0,
            g_0,
            specific_gas_constant: molar_gas_constant,
            a,
        }
    }
}

impl Atmosphere for AtmosphereIsa {
    fn pressure(&self, h: f64) -> f64 {
        let exponent = -self.g_0 / (self.a * self.specific_gas_constant);
        let t = self.temperature(h);
        (t / self.temperature_0).powf(exponent) * self.pressure_0
    }

    fn temperature(&self, h: f64) -> f64 {
        self.temperature_0 + self.a * (h - self.h_0)
    }

    fn density(&self, h: f64) -> f64 {
        let exponent = -(self.g_0 / (self.a * self.specific_gas_constant) + 1.0);
        let t = self.temperature(h);
        (t / self.temperature_0).powf(exponent) * self.density_0
    }

    fn sound_speed(&self, h: f64) -> f64 {
        331.3 * (1.0 + (self.temperature(h) - 273.15) / 273.15).sqrt()
    }
}

#[cfg(test)]
mod tests {

    use crate::utils::env::Atmosphere;

    use super::AtmosphereIsa;
    use approx::assert_relative_eq;

    #[test]
    fn test_default_isa_temperature() {
        let isa = AtmosphereIsa::default();

        assert_relative_eq!(isa.temperature(0.0), 288.15, epsilon = 0.01);
        assert_relative_eq!(isa.temperature(304.8), 286.17, epsilon = 0.01);
        assert_relative_eq!(isa.temperature(1219.2), 280.23, epsilon = 0.01);
        assert_relative_eq!(isa.temperature(4572.0), 258.43, epsilon = 0.01);
        assert_relative_eq!(isa.temperature(10668.0), 218.81, epsilon = 0.01);
    }

    #[test]
    fn test_default_isa_pressure() {
        let isa = AtmosphereIsa::default();

        assert_relative_eq!(isa.pressure(0.0), 101325.0, epsilon = 1.0);
        assert_relative_eq!(isa.pressure(304.8), 97717.0, epsilon = 1.0);
        assert_relative_eq!(isa.pressure(1219.2), 87511.0, epsilon = 1.0);
        assert_relative_eq!(isa.pressure(4572.0), 57182.0, epsilon = 1.0);
        assert_relative_eq!(isa.pressure(10668.0), 23842.0, epsilon = 1.0);
    }

    #[test]
    fn test_default_isa_density() {
        let isa = AtmosphereIsa::default();

        assert_relative_eq!(isa.density(0.0), 1.2250, epsilon = 0.0001);
        assert_relative_eq!(isa.density(304.8), 1.1896, epsilon = 0.0001);
        assert_relative_eq!(isa.density(1219.2), 1.0879, epsilon = 0.0001);
        assert_relative_eq!(isa.density(4572.0), 0.7708, epsilon = 0.0001);
        assert_relative_eq!(isa.density(10668.0), 0.3796, epsilon = 0.0001);
    }
}
