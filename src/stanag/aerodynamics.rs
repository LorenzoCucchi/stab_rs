use core::f64;
use std::f64::consts::PI;

use hdf5::File as Hdf5File;
use nalgebra::Vector3;
use ndarray::Array;
use serde_json::Value;
use std::fs::File;
use std::io::Read;

use crate::utils::env::atmosisa;

#[allow(nonstandard_style)]
pub struct Coefficients {
    alpha: ndarray::Array1<f64>,
    mach: ndarray::Array1<f64>,
    cd_fit: ndarray::Array2<f64>,
    cna_fit: ndarray::Array2<f64>,
    cma_fit: ndarray::Array2<f64>,
    cllp_fit: ndarray::Array1<f64>,
    cypa_fit: ndarray::Array1<f64>,
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
            let cllp_fit: ndarray::Array1<f64> = file.dataset("CLLP_FIT").unwrap().read().unwrap();
            let cypa_fit: ndarray::Array1<f64> = file.dataset("CYPA_FIT").unwrap().read().unwrap();
            Coefficients {
                alpha,
                mach,
                cd_fit,
                cna_fit,
                cma_fit,
                cllp_fit,
                cypa_fit,
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

            Coefficients {
                alpha,
                mach,
                cd_fit,
                cna_fit,
                cma_fit,
                cllp_fit,
                cypa_fit,
            }
        } else {
            panic!("Unsupported file format");
        }
    }
}

pub struct Aerodynamics {
    v_air_b: Vector3<f64>,
    v_norm: f64,
    roll_rate: f64,
    alpha_e: Vector3<f64>,
    rho: f64,
    diameter: f64,
    surface: f64,
    altitude: f64,
}

impl Aerodynamics {
    pub fn new(
        v_b: Vector3<f64>,
        v_wind_b: Vector3<f64>,
        roll_rate: f64,
        alpha_e: Vector3<f64>,
        diameter: f64,
        surface: f64,
        altitude: f64,
    ) -> Self {
        let v_air_b = v_b - v_wind_b;
        let v_norm = v_air_b.norm();

        Aerodynamics {
            v_air_b,
            v_norm,
            roll_rate,
            alpha_e,
            rho: 1.225,
            diameter,
            surface,
            altitude,
        }
    }

    pub fn actions(&self, c: &Coefficients) -> (Vector3<f64>, f64) {
        let (_, sound, _, _) = atmosisa(self.altitude);
        let mach = self.v_norm / sound;
        let n = c.mach.len();
        let mut low_ind = 0;
        let mut upp_ind = n - 1;

        for i in 0..n - 1 {
            if c.mach[i] <= mach && mach <= c.mach[i + 1] {
                low_ind = i;
                upp_ind = i + 1;
                break;
            }
        }

        let t = (mach - c.mach[low_ind]) / (c.mach[upp_ind] - c.mach[low_ind]);

        let cd_int: ndarray::Array1<f64> =
            &c.cd_fit.row(low_ind) + t * (&c.cd_fit.row(upp_ind) - &c.cd_fit.row(low_ind));
        let cn_int: ndarray::Array1<f64> =
            &c.cna_fit.row(low_ind) + t * (&c.cna_fit.row(upp_ind) - &c.cna_fit.row(low_ind));
        let cy_int: f64 = &c.cypa_fit[low_ind] + t * (&c.cypa_fit[upp_ind] - &c.cypa_fit[low_ind]);
        let cllp: f64 = &c.cllp_fit[low_ind] + t * (&c.cllp_fit[upp_ind] - &c.cllp_fit[low_ind]);

        let alpha: f64 = self.alpha_e.norm();
        let cd: f64 = cd_int[0] + cd_int[1] * alpha.powf(2.0) + cd_int[2] * alpha.powf(4.0);
        let lf: f64 = cn_int[0] + cn_int[1] * alpha.powf(2.0) + cn_int[2] * alpha.powf(4.0);

        let cdf: Vector3<f64> = cd * self.v_norm * self.v_air_b;
        let lff: Vector3<f64> = lf * self.v_norm.powf(2.0) * self.alpha_e;
        let mff: Vector3<f64> =
            cy_int * self.roll_rate * (self.alpha_e.cross(&self.v_air_b)) * self.diameter;

        let f: Vector3<f64> = (-cdf + lff - mff) * (PI * self.rho * self.diameter.powf(2.0) / 8.0);
        let p: f64 =
            PI * self.rho * self.diameter.powf(4.0) * self.roll_rate * cllp * self.v_norm / 8.0;

        (f, p)
    }

    pub fn calc_alphae(&self, c: &Coefficients) -> Vector3<f64> {
        let (_, sound, _, _) = atmosisa(self.altitude);
        let mach = self.v_norm / sound;
        let n = c.mach.len();
        let mut low_ind = 0;
        let mut upp_ind = n - 1;

        for i in 0..n - 1 {
            if c.mach[i] <= mach && mach <= c.mach[i + 1] {
                low_ind = i;
                upp_ind = i + 1;
                break;
            }
        }
        let t = (mach - c.mach[low_ind]) / (c.mach[upp_ind] - c.mach[low_ind]);

        let cma_int: ndarray::Array1<f64> =
            &c.cma_fit.row(low_ind) + t * (&c.cma_fit.row(upp_ind) - &c.cma_fit.row(low_ind));

        let alpha = self.alpha_e.norm();
        let cma = cma_int[0] + cma_int[1] * alpha.powf(2.0) + cma_int[2] * alpha.powf(4.0);

        let alpha_e = -8.0 * self.roll_rate * self.v_air_b
            / (PI * self.rho * self.diameter.powf(3.0) * cma * self.v_norm.powf(4.0));

        alpha_e
    }

    pub fn calc_windjump(&self, c: &Coefficients, alpha_e: Vector3<f64>) -> f64 {
        let (_, sound, _, _) = atmosisa(self.altitude);
        let mach = self.v_norm / sound;
        let n = c.mach.len();
        let mut low_ind = 0;
        let mut upp_ind = n - 1;

        for i in 0..n - 1 {
            if c.mach[i] <= mach && mach <= c.mach[i + 1] {
                low_ind = i;
                upp_ind = i + 1;
                break;
            }
        }
        let t = (mach - c.mach[low_ind]) / (c.mach[upp_ind] - c.mach[low_ind]);

        let cma_int =
            &c.cma_fit.row(low_ind) + t * (&c.cma_fit.row(upp_ind) - &c.cma_fit.row(low_ind));
        let cn_int: ndarray::Array1<f64> =
            &c.cna_fit.row(low_ind) + t * (&c.cna_fit.row(upp_ind) - &c.cna_fit.row(low_ind));

        let alpha = alpha_e.norm();
        let cma = cma_int[0] + cma_int[1] * alpha.powf(2.0) + cma_int[2] * alpha.powf(4.0);
        let cna = cn_int[0] + cn_int[1] * alpha.powf(2.0) + cn_int[2] * alpha.powf(4.0);

        return cna / (cma * self.diameter * self.v_norm.powf(2.0));
    }
}
