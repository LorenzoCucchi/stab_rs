use hdf5::File as Hdf5File;
use ndarray::Array;
use num_complex::{Complex, ComplexFloat};
use pyo3::prelude::*;
use serde_json::Value;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Read;
use std::vec;

const GRAV: f64 = 9.80665;

#[pyclass]
#[derive(Clone)]
struct Geometry {
    length: f64,
    diameter: f64,
    area: f64,
    in_x: f64,
    in_y: f64,
    xcg: f64,
    mass: f64,
    mass_grain: f64,
}

#[pymethods]
impl Geometry {
    #[new]
    fn new(length: f64, diameter: f64, in_x: f64, in_y: f64, xcg: f64, mass_grain: f64) -> Self {
        let area = PI * diameter.powf(2.0) / 4.0;
        let mass = mass_grain / 15432.4;
        Geometry {
            length,
            diameter,
            area,
            in_x,
            in_y,
            xcg,
            mass,
            mass_grain,
        }
    }

    pub fn print(&self) {
        println!("Length: {} [m]", self.length);
        println!("Diameter: {} [m]", self.diameter);
        println!("Area: {} [m2]", self.area);
        println!("Inertia X: {}", self.in_x);
        println!("Inertia Y: {}", self.in_y);
        println!("XCG: {} [m]", self.xcg);
        println!("Mass: {} [Kg]", self.mass);
        println!("Mass Grain: {} [gr]", self.mass_grain);
    }
}

#[pyclass]
#[derive(Clone)]
struct Aero {
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
    cmadcmq: f64,
    cllp: f64,
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
            let cnpa_fit: ndarray::Array2<f64> = ndarray::Array2::zeros([1, 1]);
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

    fn update_coeffs(&mut self, mach: f64) {
        self.interp1(mach);
    }

    fn get_cd(&mut self, alpha: f64) -> f64 {
        let cd = self.cd[0] + self.cd[1] * alpha.powf(2.0) + self.cd[2] * alpha.powf(4.0);
        return cd;
    }

    fn get_cna(&mut self, alpha: f64) -> f64 {
        let cn = self.cna[0] + self.cna[1] * alpha.powf(2.0) + self.cna[2] * alpha.powf(4.0);
        return cn;
    }

    fn get_cma(&mut self, alpha: f64) -> f64 {
        let cm = self.cma[0] + self.cma[1] * alpha.powf(2.0) + self.cma[2] * alpha.powf(4.0);
        return cm;
    }

    fn get_cnpa(&mut self, alpha: f64) -> f64 {
        let cnpa = self.cnpa[0] + self.cnpa[1] * alpha.powf(2.0);
        return cnpa;
    }
}

#[pyclass]
#[derive(Clone)]
struct SimData {
    #[pyo3(get, set)]
    sp_vec: Vec<f64>,
    #[pyo3(get, set)]
    vel_vec: Vec<f64>,
    #[pyo3(get, set)]
    mach_vec: Vec<f64>,
    #[pyo3(get, set)]
    p_vec: Vec<f64>,
    #[pyo3(get, set)]
    sg_vec: Vec<f64>,
    #[pyo3(get, set)]
    sd_vec: Vec<f64>,
    #[pyo3(get, set)]
    sg_lim_vec: Vec<f64>,
    #[pyo3(get, set)]
    dr_vec: Vec<f64>,
    #[pyo3(get, set)]
    alpha_vec: Vec<f64>,
    #[pyo3(get, set)]
    beta_vec: Vec<f64>,
    #[pyo3(get, set)]
    alpha_tot_vec: Vec<f64>,
    #[pyo3(get, set)]
    beta_r_vec: Vec<f64>,
    #[pyo3(get, set)]
    lambda_f_vec: Vec<f64>,
    #[pyo3(get, set)]
    lambda_s_vec: Vec<f64>,
    #[pyo3(get, set)]
    fast_freq_vec: Vec<f64>,
    #[pyo3(get, set)]
    slow_freq_vec: Vec<f64>,
    #[pyo3(get, set)]
    cd: Vec<f64>,
    #[pyo3(get, set)]
    cna: Vec<f64>,
    #[pyo3(get, set)]
    cma: Vec<f64>,
    #[pyo3(get, set)]
    cnpa: Vec<f64>,
    #[pyo3(get, set)]
    cmadcmq: Vec<f64>,
    #[pyo3(get, set)]
    cllp: Vec<f64>,
    #[pyo3(get, set)]
    cd_adim: Vec<f64>,
    #[pyo3(get, set)]
    cna_adim: Vec<f64>,
    #[pyo3(get, set)]
    cma_adim: Vec<f64>,
    #[pyo3(get, set)]
    cnpa_adim: Vec<f64>,
    #[pyo3(get, set)]
    cmadcmq_adim: Vec<f64>,
    #[pyo3(get, set)]
    cllp_adim: Vec<f64>,
}

#[pymethods]
impl SimData {
    #[new]
    fn new() -> Self {
        SimData {
            sp_vec: Vec::new(),
            vel_vec: Vec::new(),
            mach_vec: Vec::new(),
            p_vec: Vec::new(),
            sg_vec: Vec::new(),
            sd_vec: Vec::new(),
            sg_lim_vec: Vec::new(),
            dr_vec: Vec::new(),
            alpha_vec: Vec::new(),
            beta_vec: Vec::new(),
            alpha_tot_vec: Vec::new(),
            beta_r_vec: Vec::new(),
            lambda_f_vec: Vec::new(),
            lambda_s_vec: Vec::new(),
            fast_freq_vec: Vec::new(),
            slow_freq_vec: Vec::new(),
            cd: Vec::new(),
            cna: Vec::new(),
            cma: Vec::new(),
            cnpa: Vec::new(),
            cmadcmq: Vec::new(),
            cllp: Vec::new(),
            cd_adim: Vec::new(),
            cna_adim: Vec::new(),
            cma_adim: Vec::new(),
            cnpa_adim: Vec::new(),
            cmadcmq_adim: Vec::new(),
            cllp_adim: Vec::new(),
        }
    }
}

#[pyclass]
struct Simulation {
    #[pyo3(get, set)]
    geometry: Geometry,
    #[pyo3(get, set)]
    range_meters: f64,
    #[pyo3(get, set)]
    range_max: f64,
    #[pyo3(get, set)]
    twist_rate: f64,
    #[pyo3(get, set)]
    delta_yaw: f64,
    #[pyo3(get, set)]
    roll_rate: f64,
    #[pyo3(get, set)]
    temperature: f64,
    #[pyo3(get, set)]
    density: f64,
    #[pyo3(get, set)]
    sound: f64,
    #[pyo3(get, set)]
    init_vel: f64,
    #[pyo3(get, set)]
    init_mach: f64,
    #[pyo3(get, set)]
    adim: f64,
    #[pyo3(get, set)]
    iterations: usize,
    #[pyo3(get, set)]
    vec_data: SimData,
    #[pyo3(get, set)]
    aero_data: Aero,
}

#[pymethods]
impl Simulation {
    #[new]
    fn new(
        geom: Geometry,
        aero_path: String,
        range: f64,
        vel: f64,
        twist: f64,
        delta_yaw: f64,
        iter: usize,
        temp: f64,
        density: f64,
    ) -> Self {
        let range_meters = range;
        let range_max = range_meters / geom.diameter;
        let init_mach = vel / Self::sound_speed(temp);
        let roll_rate = 2.0 * PI / 0.0254 / twist * vel;
        let adim = (density * geom.area * geom.diameter) / (2.0 * geom.mass);
        Simulation {
            geometry: geom,
            range_meters,
            range_max,
            twist_rate: twist,
            delta_yaw: delta_yaw,
            roll_rate,
            temperature: temp,
            density,
            sound: Self::sound_speed(temp),
            init_vel: vel,
            init_mach,
            iterations: iter,
            adim,
            vec_data: SimData::new(),
            aero_data: Aero::new(aero_path),
        }
    }

    #[staticmethod]
    fn sound_speed(temp: f64) -> f64 {
        (1.4 * 287.05 * temp).sqrt()
    }

    fn set_conditions(
        &mut self,
        range: f64,
        vel: f64,
        twist: f64,
        delta_yaw: f64,
        iter: usize,
        temp: f64,
        density: f64,
    ) {
        self.range_meters = range;
        self.range_max = self.range_meters / self.geometry.diameter;
        self.init_vel = vel;
        self.twist_rate = twist;
        self.init_mach = vel / Self::sound_speed(temp);
        self.roll_rate = 2.0 * PI / 0.0254 / twist * vel;
        self.delta_yaw = delta_yaw;
        self.iterations = iter;
        self.temperature = temp;
        self.density = density;
        self.adim =
            (density * self.geometry.area * self.geometry.diameter) / (2.0 * self.geometry.mass);
    }

    fn init_vectors(&mut self) {
        let step = self.range_max / (self.iterations - 1) as f64;
        self.vec_data.sp_vec = vec![0.0; self.iterations];
        self.vec_data.sp_vec = (0..self.iterations).map(|i| i as f64 * step).collect();
        let len = self.vec_data.sp_vec.len();
        self.vec_data.p_vec = vec![0.0; len];
        self.vec_data.p_vec[0] = self.roll_rate;
        self.vec_data.vel_vec = vec![0.0; len];
        self.vec_data.mach_vec = vec![0.0; len];
        self.vec_data.sg_vec = vec![0.0; len];
        self.vec_data.sd_vec = vec![0.0; len];
        self.vec_data.sg_lim_vec = vec![0.0; len];
        self.vec_data.dr_vec = vec![0.0; len];
        self.vec_data.alpha_vec = vec![0.0; len];
        self.vec_data.beta_vec = vec![0.0; len];
        self.vec_data.alpha_tot_vec = vec![0.0; len];
        self.vec_data.beta_r_vec = vec![0.0; len];
        self.vec_data.lambda_f_vec = vec![0.0; len];
        self.vec_data.lambda_s_vec = vec![0.0; len];
        self.vec_data.fast_freq_vec = vec![0.0; len];
        self.vec_data.slow_freq_vec = vec![0.0; len];
        self.vec_data.cd = vec![0.0; len];
        self.vec_data.cna = vec![0.0; len];
        self.vec_data.cma = vec![0.0; len];
        self.vec_data.cnpa = vec![0.0; len];
        self.vec_data.cmadcmq = vec![0.0; len];
        self.vec_data.cllp = vec![0.0; len];
        self.vec_data.cd_adim = vec![0.0; len];
        self.vec_data.cna_adim = vec![0.0; len];
        self.vec_data.cma_adim = vec![0.0; len];
        self.vec_data.cnpa_adim = vec![0.0; len];
        self.vec_data.cmadcmq_adim = vec![0.0; len];
        self.vec_data.cllp_adim = vec![0.0; len];
    }

    fn trajectory(
        &mut self,
        p: f64,
        m: f64,
        t: f64,
        g: f64,
        h: f64,
        xi0_re: f64,
        xi0_im: f64,
        xi0_pr_re: f64,
        xi0_pr_im: f64,
        s: f64,
        d: f64,
        i: usize,
    ) {
        let help_pm24 = (p.powf(2.0) - 4.0 * m).sqrt();
        let phi_f = 0.5 * (p + help_pm24);
        let phi_s = 0.5 * (p - help_pm24);

        let lam_f = -0.5 * (h - (p * (2.0 * t - h)) / help_pm24);
        let lam_s = -0.5 * (h + (p * (2.0 * t - h)) / help_pm24);

        let xi0_prime = Complex::new(xi0_pr_re, xi0_pr_im);
        let xi0 = Complex::new(xi0_re, xi0_im);
        let j = Complex::new(0.0, 1.0);

        let kf0 = -((j * xi0_prime + phi_s * xi0) / (phi_f - phi_s));
        let ks0 = (j * xi0_prime + phi_f * xi0) / (phi_f - phi_s);

        let fast_term = kf0 * ((lam_f + j * phi_f) * s).exp();
        let slow_term = ks0 * ((lam_s + j * phi_s) * s).exp();

        let xi = fast_term + slow_term + j * ((p * g) / (m + j * p * t));

        self.vec_data.beta_vec[i] = xi.im();
        self.vec_data.alpha_vec[i] = xi.re();
        self.vec_data.alpha_tot_vec[i] = (xi.im().powf(2.0) + xi.re().powf(2.0)).sqrt();
        self.vec_data.beta_r_vec[i] = ((p * g) / (m + j * p * t)).re();
        self.vec_data.slow_freq_vec[i] = (phi_s * self.vec_data.vel_vec[i]) / (2.0 * PI * d);
        self.vec_data.fast_freq_vec[i] = (phi_f * self.vec_data.vel_vec[i]) / (2.0 * PI * d);
        self.vec_data.lambda_s_vec[i] = lam_s.re();
        self.vec_data.lambda_f_vec[i] = lam_f.re();
    }

    fn run(&mut self) {
        self.init_vectors();
        let j = Complex::new(0.0, 1.0);
        let kx_2: f64 = self.geometry.mass * self.geometry.diameter.powf(2.0) / self.geometry.in_x;
        let ky_2: f64 = self.geometry.mass * self.geometry.diameter.powf(2.0) / self.geometry.in_y;
        let inx_iny = self.geometry.in_x / self.geometry.in_y;
        let diam = self.geometry.diameter;

        self.update_aero(0, 0.0);

        let mut P = inx_iny * ((self.vec_data.p_vec[0] * diam) / self.vec_data.vel_vec[0]);
        let mut M = ky_2 * self.vec_data.cma[0];
        let mut T = self.vec_data.cna[0];
        let mut G = GRAV * diam * 0.0_f64.cos() / self.vec_data.vel_vec[0].powf(2.0);
        let mut H = self.vec_data.cna[0] - self.vec_data.cd[0] - ky_2 * self.vec_data.cmadcmq[0];

        self.vec_data.sg_vec[0] = P.powf(2.0) / (4.0 * M);
        self.vec_data.sd_vec[0] = 2.0 * T / H;
        self.vec_data.sg_lim_vec[0] =
            1.0 / (self.vec_data.p_vec[0] * (2.0 - self.vec_data.sd_vec[0]));

        let eps = (1.0 - 1.0 / self.vec_data.sg_vec[0]).sqrt()
            * (self.delta_yaw.to_radians().sin())
            / (2.0 * (1.0 / inx_iny) - 1.0);
        println!("{:?}", eps);
        let xi0 = eps.sin() * j.exp();
        let xi0_prime = j * ((self.roll_rate * diam) / self.init_vel) * xi0;

        let mut delta_s = 0.0;
        let mut ttime = 0.0;
        let mut kp = 0.0;
        let mut s = 0.0;
        for i in 0..self.vec_data.sp_vec.len() {
            s = self.vec_data.sp_vec[i];
            if i == 0 {
                self.vec_data.vel_vec[i] = self.init_vel;
                self.update_aero(i, self.vec_data.alpha_tot_vec[0]);
            } else {
                self.vec_data.vel_vec[i] = self.vec_data.vel_vec[0]
                    * (-trapz(&self.vec_data.cd_adim[0..i], &self.vec_data.sp_vec[0..i])).exp();

                delta_s = self.vec_data.sp_vec[i] - self.vec_data.sp_vec[i - 1];
                ttime = ttime + delta_s * diam / self.vec_data.vel_vec[i];

                kp = -(kx_2 * self.vec_data.cllp_adim[i - 1] + self.vec_data.cd_adim[i - 1]);

                self.vec_data.p_vec[i] = self.vec_data.vel_vec[i] * self.vec_data.p_vec[0]
                    / self.vec_data.vel_vec[0]
                    * (-kp * self.vec_data.sp_vec[i]).exp();

                self.update_aero(i, self.vec_data.alpha_tot_vec[i - 1]);
            }

            P = inx_iny * ((self.vec_data.p_vec[i] * diam) / self.vec_data.vel_vec[i]);
            M = ky_2 * self.vec_data.cma_adim[i];
            T = self.vec_data.cna_adim[i] + kx_2 * self.vec_data.cnpa_adim[i];
            G = GRAV * diam * 0.0_f64.cos() / self.vec_data.vel_vec[i].powf(2.0);
            H = self.vec_data.cna_adim[i]
                - self.vec_data.cd_adim[i]
                - ky_2 * self.vec_data.cmadcmq_adim[i];

            self.vec_data.sg_vec[i] = P.powf(2.0) / (4.0 * M);
            self.vec_data.sd_vec[i] = 2.0 * T / H;
            self.vec_data.sg_lim_vec[i] =
                1.0 / (self.vec_data.sd_vec[i] * (2.0 - self.vec_data.sd_vec[i]));

            self.vec_data.dr_vec[i] = (j / ky_2
                * (self.vec_data.p_vec[i] * G / 2.0)
                * (self.vec_data.cna_adim[i] / self.vec_data.cma_adim[i])
                * s.powf(2.0)
                * (1.0
                    + 2.0 / 3.0 * (self.vec_data.cd_adim[i] * s)
                    + 1.0 / 3.0 * (self.vec_data.cd_adim[i] * s).powf(2.0)))
            .re();
            self.trajectory(
                P,
                M,
                T,
                G,
                H,
                xi0.re(),
                xi0.im(),
                xi0_prime.re(),
                xi0_prime.im(),
                s,
                diam,
                i,
            );
        }
    }

    fn update_aero(&mut self, ind: usize, alpha: f64) {
        self.vec_data.mach_vec[ind] = self.vec_data.vel_vec[ind] / self.sound;

        self.aero_data.update_coeffs(self.vec_data.mach_vec[ind]);

        self.vec_data.cd[ind] = self.aero_data.get_cd(alpha);
        self.vec_data.cna[ind] = self.aero_data.get_cna(alpha);
        self.vec_data.cma[ind] = self.aero_data.get_cma(alpha);
        self.vec_data.cnpa[ind] = self.aero_data.get_cnpa(alpha);
        self.vec_data.cmadcmq[ind] = self.aero_data.cmadcmq;
        self.vec_data.cllp[ind] = self.aero_data.cllp;

        self.vec_data.cd_adim[ind] = self.vec_data.cd[ind] * self.adim;
        self.vec_data.cna_adim[ind] = self.vec_data.cna[ind] * self.adim;
        self.vec_data.cma_adim[ind] = self.vec_data.cma[ind] * self.adim;
        self.vec_data.cnpa_adim[ind] = self.vec_data.cnpa[ind] * self.adim;
        self.vec_data.cmadcmq_adim[ind] = self.vec_data.cmadcmq[ind] * self.adim;
        self.vec_data.cllp_adim[ind] = self.vec_data.cllp[ind] * self.adim;
    }

    fn write_file(&self) {
        let file = Hdf5File::create("sim_res.hdf5").unwrap();

        file.new_dataset::<f64>()
            .shape(self.vec_data.sp_vec.len())
            .create("sp_vec")
            .unwrap()
            .write(&self.vec_data.sp_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.vel_vec.len())
            .create("vel_vec")
            .unwrap()
            .write(&self.vec_data.vel_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.mach_vec.len())
            .create("mach_vec")
            .unwrap()
            .write(&self.vec_data.mach_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.p_vec.len())
            .create("p_vec")
            .unwrap()
            .write(&self.vec_data.p_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.sg_vec.len())
            .create("sg_vec")
            .unwrap()
            .write(&self.vec_data.sg_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.sd_vec.len())
            .create("sd_vec")
            .unwrap()
            .write(&self.vec_data.sd_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.sg_lim_vec.len())
            .create("sg_lim_vec")
            .unwrap()
            .write(&self.vec_data.sg_lim_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.dr_vec.len())
            .create("dr_vec")
            .unwrap()
            .write(&self.vec_data.dr_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.alpha_vec.len())
            .create("alpha_vec")
            .unwrap()
            .write(&self.vec_data.alpha_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.beta_vec.len())
            .create("beta_vec")
            .unwrap()
            .write(&self.vec_data.beta_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.alpha_tot_vec.len())
            .create("alpha_tot_vec")
            .unwrap()
            .write(&self.vec_data.alpha_tot_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.beta_r_vec.len())
            .create("beta_r_vec")
            .unwrap()
            .write(&self.vec_data.beta_r_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.lambda_f_vec.len())
            .create("lambda_f_vec")
            .unwrap()
            .write(&self.vec_data.lambda_f_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.lambda_s_vec.len())
            .create("lambda_s_vec")
            .unwrap()
            .write(&self.vec_data.lambda_s_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.fast_freq_vec.len())
            .create("fast_freq_vec")
            .unwrap()
            .write(&self.vec_data.fast_freq_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.slow_freq_vec.len())
            .create("slow_freq_vec")
            .unwrap()
            .write(&self.vec_data.slow_freq_vec)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.cd.len())
            .create("cd")
            .unwrap()
            .write(&self.vec_data.cd)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.cna.len())
            .create("cna")
            .unwrap()
            .write(&self.vec_data.cna)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.cma.len())
            .create("cma")
            .unwrap()
            .write(&self.vec_data.cma)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.cmadcmq.len())
            .create("cmadcmq")
            .unwrap()
            .write(&self.vec_data.cmadcmq)
            .unwrap();
        file.new_dataset::<f64>()
            .shape(self.vec_data.cllp.len())
            .create("cllp")
            .unwrap()
            .write(&self.vec_data.cllp)
            .unwrap();
    }
}

fn trapz(y: &[f64], x: &[f64]) -> f64 {
    y.iter()
        .zip(x.iter().skip(1))
        .enumerate()
        .map(|(i, (&y_c, &x_n))| {
            let x_c = x[i];
            (x_n - x_c) * (y[i + 1] + y_c) / 2.0
        })
        .sum()
}

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn stab_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Geometry>()?;
    m.add_class::<Simulation>()?;
    m.add_class::<Aero>()?;
    m.add_class::<SimData>()?;
    Ok(())
}
