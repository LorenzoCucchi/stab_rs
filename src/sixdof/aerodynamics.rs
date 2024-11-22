use core::f64;
use std::f64::consts::PI;

use hdf5::File as Hdf5File;
use nalgebra::{vector, Vector3};
use ndarray::Array;
use serde_json::Value;
use std::fs::File;
use std::io::Read;

use super::atmosphere::Atmosphere;

#[allow(nonstandard_style, non_snake_case)]
pub struct Coefficients {
    alpha: ndarray::Array1<f64>,
    mach: ndarray::Array1<f64>,
    CA_fit: ndarray::Array2<f64>,
    CNa_fit: ndarray::Array2<f64>,
    CMa_fit: ndarray::Array2<f64>,
    cmadcmq_fit: ndarray::Array1<f64>,
    CLLp_fit: ndarray::Array1<f64>,
    CYpa_fit: ndarray::Array1<f64>,
    Cnp1_fit: ndarray::Array1<f64>,
    Cnp3_fit: ndarray::Array1<f64>,
}

impl Coefficients {
    pub fn from_path(path: &str) -> Self {
        let is_hdf5 = path.ends_with(".h5") || path.ends_with(".hdf5");
        let is_json = path.ends_with(".json");

        if is_hdf5 {
            let file = Hdf5File::open(&path).expect("Failed to open HDF5 file");

            let alpha: ndarray::Array1<f64> = file.dataset("alpha").unwrap().read_1d().unwrap();
            let mach: ndarray::Array1<f64> = file.dataset("mach").unwrap().read().unwrap();
            let cd_fit: ndarray::Array2<f64> = file.dataset("CD_FIT").unwrap().read().unwrap();
            let cna_fit: ndarray::Array2<f64> = file.dataset("CNA_FIT").unwrap().read().unwrap();
            let cma_fit: ndarray::Array2<f64> = file.dataset("CMA_FIT").unwrap().read().unwrap();
            let cmadcmq: ndarray::Array1<f64> = file.dataset("CMADCMQ").unwrap().read().unwrap();
            let cllp_fit: ndarray::Array1<f64> = file.dataset("CLLP_FIT").unwrap().read().unwrap();
            let cypa_fit: ndarray::Array1<f64> = file.dataset("CYPA_FIT").unwrap().read().unwrap();
            let cnp1_fit: ndarray::Array1<f64> = file.dataset("CNP1_FIT").unwrap().read().unwrap();
            let cnp3_fit: ndarray::Array1<f64> = file.dataset("CNP3_FIT").unwrap().read().unwrap();
            Coefficients {
                alpha,
                mach,
                CA_fit: cd_fit,
                CNa_fit: cna_fit,
                CMa_fit: cma_fit,
                cmadcmq_fit: cmadcmq,
                CLLp_fit: cllp_fit,
                CYpa_fit: cypa_fit,
                Cnp1_fit: cnp1_fit,
                Cnp3_fit: cnp3_fit,
            }
        } else if is_json {
            let mut file = File::open(&path).expect("Failed to open JSON file");
            let mut content = String::new();
            file.read_to_string(&mut content)
                .expect("Failed to read JSON file");

            let json_data: Value = serde_json::from_str(&content).expect("Failed to parse JSON");

            let mach_values: Vec<f64> = json_data["mach"]
                .as_array()
                .unwrap()
                .iter()
                .map(|val| val.as_f64().unwrap())
                .collect();

            let mach = Array::from_vec(mach_values.clone());

            let alpha = Array::from_vec(
                json_data["alpha"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|val| val.as_f64().unwrap())
                    .collect(),
            );

            let cd_fit = Array::from_shape_vec(
                (mach_values.len(), alpha.len_of(ndarray::Axis(0))),
                json_data["cd_fit"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .collect(),
            )
            .unwrap();

            let cna_fit = Array::from_shape_vec(
                (mach_values.len(), alpha.len_of(ndarray::Axis(0))),
                json_data["cna_fit"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .collect(),
            )
            .unwrap();

            let cma_fit = Array::from_shape_vec(
                (mach_values.len(), alpha.len_of(ndarray::Axis(0))),
                json_data["cma_fit"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .collect(),
            )
            .unwrap();

            let cmadcmq = Array::from_vec(
                json_data["cmadcmq_fit"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
            );

            let cllp_fit = Array::from_vec(
                json_data["cllp_fit"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
            );

            let cypa_fit = Array::from_vec(
                json_data["cypa_fit"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
            );

            let cnp1_fit = Array::from_vec(
                json_data["cnp1_fit"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
            );

            let cnp3_fit = Array::from_vec(
                json_data["cnp3_fit"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
            );
            Coefficients {
                alpha,
                mach,
                CA_fit: cd_fit,
                CNa_fit: cna_fit,
                CMa_fit: cma_fit,
                cmadcmq_fit: cmadcmq,
                CLLp_fit: cllp_fit,
                CYpa_fit: cypa_fit,
                Cnp1_fit: cnp1_fit,
                Cnp3_fit: cnp3_fit,
            }
        } else {
            panic!("Unsupported file format");
        }
    }
}

const V_SMALL: f64 = 1.0e-5;

pub struct AeroState {
    pub v_air_b: Vector3<f64>,
    pub v_norm: f64,
    pub w: Vector3<f64>,
    pub h: f64,
}

impl AeroState {
    pub fn new(v_b: Vector3<f64>, v_wind_b: Vector3<f64>, w: Vector3<f64>, h: f64) -> AeroState {
        let v_air_b = v_b - v_wind_b;
        let v_norm = v_air_b.norm();

        AeroState {
            v_air_b,
            v_norm,
            w,
            h,
        }
    }
}

pub struct AerodynamicsResult {
    pub alpha: f64,
    pub beta: f64,
    pub forces: Vector3<f64>,
    pub moments: Vector3<f64>,
}

pub struct Aerodynamics {
    atmosphere: Box<dyn Atmosphere + Send>,
    diameter: f64,
    surface: f64,
    pub coefficients: Coefficients,
}

impl Aerodynamics {
    pub fn new(
        diameter: f64,
        surface: f64,
        atmosphere: Box<dyn Atmosphere + Send>,
        coefficients: Coefficients,
    ) -> Self {
        Aerodynamics {
            atmosphere,
            diameter,
            surface,
            coefficients,
        }
    }

    pub fn alpha(&self, state: &AeroState) -> f64 {
        if state.v_air_b[0].abs() >= V_SMALL {
            f64::atan(state.v_air_b[2] / state.v_air_b[0])
        } else if state.v_air_b[2].abs() >= V_SMALL {
            f64::consts::FRAC_PI_2 * state.v_air_b[2].signum()
        } else {
            0.0
        }
    }

    pub fn beta(&self, state: &AeroState) -> f64 {
        if state.v_norm >= V_SMALL {
            f64::asin(state.v_air_b[1] / state.v_norm)
        } else if state.v_air_b[1].abs() >= V_SMALL {
            f64::consts::FRAC_PI_2 * state.v_air_b[1].signum()
        } else {
            0.0
        }
    }

    pub fn calc(&self, state: &AeroState) -> AerodynamicsResult {
        let alpha = self.alpha(state);
        let beta = self.beta(state);

        let (forces, moments) = self.actions(alpha, beta, state);

        AerodynamicsResult {
            alpha,
            beta,
            forces,
            moments,
        }
    }

    #[allow(non_snake_case)]
    pub fn actions(
        &self,
        alpha: f64,
        beta: f64,
        state: &AeroState,
    ) -> (Vector3<f64>, Vector3<f64>) {
        let sound = self.atmosphere.sound_speed(state.h);
        let mach = state.v_norm / sound;

        let (low_ind, upp_ind, t) = self
            .coefficients
            .mach
            .windows(2)
            .into_iter()
            .enumerate()
            .find_map(|(i, pair)| {
                if pair[0] <= mach && mach <= pair[1] {
                    let t = (mach - pair[0]) / (pair[1] - pair[0]);
                    Some((i, i + 1, t))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                panic!(
                    "Error: The mach value {} is outside the valid range of coefficients.",
                    mach
                );
            });

        if cfg!(debug_assertions) {
            println!("Low index: {}, Upper index: {}, t: {}", low_ind, upp_ind, t);
        }

        let ca_int: ndarray::Array1<f64> = &self.coefficients.CA_fit.row(low_ind)
            + t * (&self.coefficients.CA_fit.row(upp_ind) - &self.coefficients.CA_fit.row(low_ind));
        let cn_int: ndarray::Array1<f64> = &self.coefficients.CNa_fit.row(low_ind)
            + t * (&self.coefficients.CNa_fit.row(upp_ind)
                - &self.coefficients.CNa_fit.row(low_ind));
        let cm_int: ndarray::Array1<f64> = &self.coefficients.CMa_fit.row(low_ind)
            + t * (&self.coefficients.CMa_fit.row(upp_ind)
                - &self.coefficients.CMa_fit.row(low_ind));
        let cmqcmad = &self.coefficients.cmadcmq_fit[low_ind]
            + t * (&self.coefficients.cmadcmq_fit[upp_ind]
                - &self.coefficients.cmadcmq_fit[low_ind]);
        let cypa_int: f64 = &self.coefficients.CYpa_fit[low_ind]
            + t * (&self.coefficients.CYpa_fit[upp_ind] - &self.coefficients.CYpa_fit[low_ind]);
        let cllp = &self.coefficients.CLLp_fit[low_ind]
            + t * (&self.coefficients.CLLp_fit[upp_ind] - &self.coefficients.CLLp_fit[low_ind]);
        let cnp1 = &self.coefficients.Cnp1_fit[low_ind]
            + t * (&self.coefficients.Cnp1_fit[upp_ind] - &self.coefficients.Cnp1_fit[low_ind]);
        let cnp3 = &self.coefficients.Cnp3_fit[low_ind]
            + t * (&self.coefficients.Cnp3_fit[upp_ind] - &self.coefficients.Cnp3_fit[low_ind]);

        let ca = (ca_int[0] + ca_int[1] * alpha.powf(2.0) + ca_int[2] * alpha.powf(4.0))
            * state.v_air_b[0].signum();
        let cna = (cn_int[0] + cn_int[1] * alpha.powf(2.0) + cn_int[2] * alpha.powf(4.0))
            * state.v_air_b[0].signum();
        let cma = (cm_int[0] + cm_int[1] * alpha.powf(2.0) + cm_int[2] * alpha.powf(4.0))
            * state.v_air_b[0].signum();

        let q_v = 0.5 * self.atmosphere.density(state.h) * state.v_norm;
        let q = q_v * state.v_norm;

        let f = vector![
            -q * self.surface * ca, /* x-axis */
            -q * self.surface * cna * beta /*- q * self.surface * cypa_int * alpha.sin() * state.w[0]*/, /* y-axis cn=-cy */
            -q * self.surface * cna * alpha /*- q * self.surface * cypa_int * beta.sin() * state.w[0]*/, /* z-axis */
        ];

        let t = vector![
            0.5 * q_v * self.surface * self.diameter.powf(2.0) * cllp * state.w[0],
            q * self.surface * cma * alpha * self.diameter
                + 0.5 * q_v * self.surface * self.diameter.powf(2.0) * cmqcmad * state.w[1]
                + 0.5
                    * q_v
                    * self.surface
                    * self.diameter.powf(2.0)
                    * (cnp1 * beta.sin() + cnp3 * (beta.sin()).powf(3.0))
                    * state.w[0],
            -q * self.surface * cma * beta * self.diameter
                + 0.5 * q_v * self.surface * self.diameter.powf(2.0) * cmqcmad * state.w[2]
                - 0.5
                    * q_v
                    * self.surface
                    * self.diameter.powf(2.0)
                    * (cnp1 * alpha.sin() + cnp3 * (alpha.sin()).powf(3.0))
                    * state.w[0],
        ];

        (f, t)
    }
}
