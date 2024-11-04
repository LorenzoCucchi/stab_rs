use hdf5::File as Hdf5File;
use ndarray::Array;
use pyo3::prelude::*;
use serde_json::Value;
use std::fs::File;
use std::io::Read;

#[allow(unused)]
#[pyclass]
#[derive(Clone)]
pub struct Aero {
    path: String,
    alpha: ndarray::Array1<f64>,
    mach: ndarray::Array1<f64>,
    cd_fit: ndarray::Array2<f64>,
    cna_fit: ndarray::Array2<f64>,
    cma_fit: ndarray::Array2<f64>,
    cmadcmq_fit: ndarray::Array1<f64>,
    cnpa_fit: ndarray::Array2<f64>,
    cllp_fit: ndarray::Array1<f64>,
    cd: ndarray::Array1<f64>,
    cna: ndarray::Array1<f64>,
    cma: ndarray::Array1<f64>,
    cnpa: ndarray::Array1<f64>,
    pub cmadcmq: f64,
    pub cllp: f64,
}

#[pymethods]
impl Aero {
    #[new]
    pub fn new(path: String) -> Self {
        let is_hdf5 = path.ends_with(".h5") || path.ends_with(".hdf5");
        let is_json = path.ends_with(".json");

        if is_hdf5 {
            let file = Hdf5File::open(&path).expect("Failed to open HDF5 file");

            let alpha: ndarray::Array1<f64> = file.dataset("alpha").unwrap().read_1d().unwrap();
            let mach: ndarray::Array1<f64> = file.dataset("mach").unwrap().read().unwrap();
            let cd_fit: ndarray::Array2<f64> = file.dataset("CD_FIT").unwrap().read().unwrap();
            let cna_fit: ndarray::Array2<f64> = file.dataset("CNA_FIT").unwrap().read().unwrap();
            let cma_fit: ndarray::Array2<f64> = file.dataset("CMA_FIT").unwrap().read().unwrap();
   
            let cnpa_fit: ndarray::Array2<f64> = match file.dataset("CNPA_FIT") {
                Ok(dataset) => dataset.read().unwrap(),
                Err(_) => ndarray::Array2::zeros((cma_fit.dim().0, cma_fit.dim().1)),
            };
            let cmadcmq_fit: ndarray::Array1<f64> =
                file.dataset("CMADCMQ").unwrap().read().unwrap();
            let cllp_fit: ndarray::Array1<f64> = file.dataset("CLLP_FIT").unwrap().read().unwrap();

            Aero {
                path,
                alpha,
                mach,
                cd_fit,
                cna_fit,
                cma_fit,
                cnpa_fit,
                cmadcmq_fit,
                cllp_fit,
                cd: Array::zeros(3),
                cna: Array::zeros(3),
                cma: Array::zeros(3),
                cnpa: Array::zeros(2),
                cmadcmq: 0.0,
                cllp: 0.0,
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

            let cnpa_fit = Array::from_shape_vec(
                (mach_values.len(), alpha.len_of(ndarray::Axis(0))),
                json_data["cnpa_fit"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .collect(),
            )
            .unwrap();

            let cmadcmq_fit = Array::from_vec(
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

            Aero {
                path,
                alpha,
                mach,
                cd_fit,
                cna_fit,
                cma_fit,
                cnpa_fit,
                cmadcmq_fit,
                cllp_fit,
                cd: Array::zeros(3),
                cna: Array::zeros(3),
                cma: Array::zeros(3),
                cnpa: Array::zeros(3),
                cmadcmq: 0.0,
                cllp: 0.0,
            }
        } else {
            panic!("Unsupported file format");
        }
    }

    fn interp1(&mut self, mach: f64) {
        let n = self.mach.len();
        let mut low_ind = 0;
        let mut upp_ind = n - 1;

        for i in 0..n - 1 {
            if self.mach[i] <= mach && mach <= self.mach[i + 1] {
                low_ind = i;
                upp_ind = i + 1;
                break;
            }
        }

        let t = (mach - self.mach[low_ind]) / (self.mach[upp_ind] - self.mach[low_ind]);

        self.cd =
            &self.cd_fit.row(low_ind) + t * (&self.cd_fit.row(upp_ind) - &self.cd_fit.row(low_ind));
        self.cna = &self.cna_fit.row(low_ind)
            + t * (&self.cna_fit.row(upp_ind) - &self.cna_fit.row(low_ind));
        self.cma = &self.cma_fit.row(low_ind)
            + t * (&self.cma_fit.row(upp_ind) - &self.cma_fit.row(low_ind));
        self.cnpa = &self.cnpa_fit.row(low_ind)
            + t * (&self.cnpa_fit.row(upp_ind) - &self.cnpa_fit.row(low_ind));
        self.cmadcmq = &self.cmadcmq_fit[low_ind]
            + t * (&self.cmadcmq_fit[upp_ind] - &self.cmadcmq_fit[low_ind]);
        self.cllp =
            &self.cllp_fit[low_ind] + t * (&self.cllp_fit[upp_ind] - &self.cllp_fit[low_ind]);
    }

    pub fn update_coeffs(&mut self, mach: f64) {
        self.interp1(mach);
    }

    pub fn get_cd(&mut self, alpha: f64) -> f64 {
        let cd = self.cd[0] + self.cd[1] * alpha.powf(2.0) + self.cd[2] * alpha.powf(4.0);
        return cd;
    }

    pub fn get_cna(&mut self, alpha: f64) -> f64 {
        let cn = self.cna[0] + self.cna[1] * alpha.powf(2.0) + self.cna[2] * alpha.powf(4.0);
        return cn;
    }

    pub fn get_cma(&mut self, alpha: f64) -> f64 {
        let cm = self.cma[0] + self.cma[1] * alpha.powf(2.0) + self.cma[2] * alpha.powf(4.0);
        return cm;
    }

    pub fn get_cnpa(&mut self, alpha: f64) -> f64 {
        let cnpa = self.cnpa[0] + self.cnpa[1] * alpha.powf(2.0);
        return cnpa;
    }
}
