use hdf5::File as Hdf5File;
use num_complex::{Complex, ComplexFloat};
use pyo3::prelude::*;
use std::f64::consts::PI;
use std::vec;

mod aero;
mod geometry;

use aero::Aero;
use geometry::Geometry;
const GRAV: f64 = 9.80665;

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
        let j: Complex<f64> = Complex::new(0.0, 1.0);
        let kx_2: f64 = self.geometry.mass * self.geometry.diameter.powf(2.0) / self.geometry.in_x;
        let ky_2: f64 = self.geometry.mass * self.geometry.diameter.powf(2.0) / self.geometry.in_y;
        let inx_iny: f64 = self.geometry.in_x / self.geometry.in_y;
        let diam: f64 = self.geometry.diameter;

        self.update_aero(0, 0.0);

        let mut _p: f64 = inx_iny * ((self.vec_data.p_vec[0] * diam) / self.vec_data.vel_vec[0]);
        let mut _m: f64 = ky_2 * self.vec_data.cma[0];
        let mut _t: f64 = self.vec_data.cna[0];
        let mut _g: f64 = GRAV * diam * 0.0_f64.cos() / self.vec_data.vel_vec[0].powf(2.0);
        let mut _h: f64 =
            self.vec_data.cna[0] - self.vec_data.cd[0] - ky_2 * self.vec_data.cmadcmq[0];

        self.vec_data.sg_vec[0] = _p.powf(2.0) / (4.0 * _m);
        self.vec_data.sd_vec[0] = 2.0 * _t / _h;
        self.vec_data.sg_lim_vec[0] =
            1.0 / (self.vec_data.p_vec[0] * (2.0 - self.vec_data.sd_vec[0]));

        let eps: f64 = (1.0 - 1.0 / self.vec_data.sg_vec[0]).sqrt()
            * (self.delta_yaw.to_radians().sin())
            / (2.0 * (1.0 / inx_iny) - 1.0);

        let xi0 = eps.sin() * j.exp();
        let xi0_prime = j * ((self.roll_rate * diam) / self.init_vel) * xi0;

        let mut ttime = 0.0;
        for i in 0..self.vec_data.sp_vec.len() {
            let s = self.vec_data.sp_vec[i];
            if i == 0 {
                self.vec_data.vel_vec[i] = self.init_vel;
                self.update_aero(i, self.vec_data.alpha_tot_vec[0]);
            } else {
                self.vec_data.vel_vec[i] = self.vec_data.vel_vec[0]
                    * (-trapz(&self.vec_data.cd_adim[0..i], &self.vec_data.sp_vec[0..i])).exp();

                let delta_s = self.vec_data.sp_vec[i] - self.vec_data.sp_vec[i - 1];
                ttime = ttime + delta_s * diam / self.vec_data.vel_vec[i];

                let kp = -(kx_2 * self.vec_data.cllp_adim[i - 1] + self.vec_data.cd_adim[i - 1]);

                self.vec_data.p_vec[i] = self.vec_data.vel_vec[i] * self.vec_data.p_vec[0]
                    / self.vec_data.vel_vec[0]
                    * (-kp * self.vec_data.sp_vec[i]).exp();

                self.update_aero(i, self.vec_data.alpha_tot_vec[i - 1]);
            }

            _p = inx_iny * ((self.vec_data.p_vec[i] * diam) / self.vec_data.vel_vec[i]);
            _m = ky_2 * self.vec_data.cma_adim[i];
            _t = self.vec_data.cna_adim[i] + kx_2 * self.vec_data.cnpa_adim[i];
            _g = GRAV * diam * 0.0_f64.cos() / self.vec_data.vel_vec[i].powf(2.0);
            _h = self.vec_data.cna_adim[i]
                - self.vec_data.cd_adim[i]
                - ky_2 * self.vec_data.cmadcmq_adim[i];

            self.vec_data.sg_vec[i] = _p.powf(2.0) / (4.0 * _m);
            self.vec_data.sd_vec[i] = 2.0 * _t / _h;
            self.vec_data.sg_lim_vec[i] =
                1.0 / (self.vec_data.sd_vec[i] * (2.0 - self.vec_data.sd_vec[i]));

            self.vec_data.dr_vec[i] = (j / ky_2
                * (self.vec_data.p_vec[i] * _g / 2.0)
                * (self.vec_data.cna_adim[i] / self.vec_data.cma_adim[i])
                * s.powf(2.0)
                * (1.0
                    + 2.0 / 3.0 * (self.vec_data.cd_adim[i] * s)
                    + 1.0 / 3.0 * (self.vec_data.cd_adim[i] * s).powf(2.0)))
            .re();
            self.trajectory(
                _p,
                _m,
                _t,
                _g,
                _h,
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

/// stab_rs module
#[pymodule]
fn stab_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Geometry>()?;
    m.add_class::<Simulation>()?;
    m.add_class::<Aero>()?;
    m.add_class::<SimData>()?;
    Ok(())
}
