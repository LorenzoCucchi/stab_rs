use hdf5::File as Hdf5File;
use num_complex::{Complex, ComplexFloat};
use pyo3::prelude::*;
use std::f64::consts::PI;
use std::vec;
use struct_iterable::Iterable;

mod aero;
mod geometry;

use aero::Aero;
use geometry::Geometry;
const GRAV: f64 = 9.80665;

#[pyclass]
#[derive(Clone, Iterable, Debug)]
struct NamedVec {
    #[pyo3(get)]
    name: &'static str,
    #[pyo3(get, set)]
    data: Vec<f64>,
}

macro_rules! write_datasets {
    ($file: expr, $vec_data:expr) => {
        for (_, dat) in $vec_data.iter() {
            let named_vec = dat.downcast_ref::<NamedVec>().unwrap();
            $file
                .new_dataset::<f64>()
                .shape(named_vec.data.len())
                .create(named_vec.name)
                .unwrap()
                .write(&named_vec.data)
                .unwrap();
        }
    };
}

#[pyclass]
#[derive(Clone, Iterable)]
struct SimData {
    #[pyo3(get, set)]
    sp_vec: NamedVec,
    #[pyo3(get, set)]
    vel_vec: NamedVec,
    #[pyo3(get, set)]
    mach_vec: NamedVec,
    #[pyo3(get, set)]
    p_vec: NamedVec,
    #[pyo3(get, set)]
    sg_vec: NamedVec,
    #[pyo3(get, set)]
    sd_vec: NamedVec,
    #[pyo3(get, set)]
    sg_lim_vec: NamedVec,
    #[pyo3(get, set)]
    dr_vec: NamedVec,
    #[pyo3(get, set)]
    alpha_vec: NamedVec,
    #[pyo3(get, set)]
    beta_vec: NamedVec,
    #[pyo3(get, set)]
    alpha_tot_vec: NamedVec,
    #[pyo3(get, set)]
    beta_r_vec: NamedVec,
    #[pyo3(get, set)]
    lambda_f_vec: NamedVec,
    #[pyo3(get, set)]
    lambda_s_vec: NamedVec,
    #[pyo3(get, set)]
    fast_freq_vec: NamedVec,
    #[pyo3(get, set)]
    slow_freq_vec: NamedVec,
    #[pyo3(get, set)]
    cd: NamedVec,
    #[pyo3(get, set)]
    cna: NamedVec,
    #[pyo3(get, set)]
    cma: NamedVec,
    #[pyo3(get, set)]
    cnpa: NamedVec,
    #[pyo3(get, set)]
    cmadcmq: NamedVec,
    #[pyo3(get, set)]
    cllp: NamedVec,
    #[pyo3(get, set)]
    cd_adim: NamedVec,
    #[pyo3(get, set)]
    cna_adim: NamedVec,
    #[pyo3(get, set)]
    cma_adim: NamedVec,
    #[pyo3(get, set)]
    cnpa_adim: NamedVec,
    #[pyo3(get, set)]
    cmadcmq_adim: NamedVec,
    #[pyo3(get, set)]
    cllp_adim: NamedVec,
}

#[pymethods]
impl SimData {
    #[new]
    fn new() -> Self {
        SimData {
            sp_vec: NamedVec {
                name: "sp_vec",
                data: Vec::new(),
            },
            vel_vec: NamedVec {
                name: "vel_vec",
                data: Vec::new(),
            },
            mach_vec: NamedVec {
                name: "mach_vec",
                data: Vec::new(),
            },
            p_vec: NamedVec {
                name: "p_vec",
                data: Vec::new(),
            },
            sg_vec: NamedVec {
                name: "sg_vec",
                data: Vec::new(),
            },
            sd_vec: NamedVec {
                name: "sd_vec",
                data: Vec::new(),
            },
            sg_lim_vec: NamedVec {
                name: "sg_lim_vec",
                data: Vec::new(),
            },
            dr_vec: NamedVec {
                name: "dr_vec",
                data: Vec::new(),
            },
            alpha_vec: NamedVec {
                name: "alpha_vec",
                data: Vec::new(),
            },
            beta_vec: NamedVec {
                name: "beta_vec",
                data: Vec::new(),
            },
            alpha_tot_vec: NamedVec {
                name: "alpha_tot_vec",
                data: Vec::new(),
            },
            beta_r_vec: NamedVec {
                name: "beta_r_vec",
                data: Vec::new(),
            },
            lambda_f_vec: NamedVec {
                name: "lambda_f_vec",
                data: Vec::new(),
            },
            lambda_s_vec: NamedVec {
                name: "lambda_s_vec",
                data: Vec::new(),
            },
            fast_freq_vec: NamedVec {
                name: "fast_freq_vec",
                data: Vec::new(),
            },
            slow_freq_vec: NamedVec {
                name: "slow_freq_vec",
                data: Vec::new(),
            },
            cd: NamedVec {
                name: "cd_vec",
                data: Vec::new(),
            },
            cna: NamedVec {
                name: "cna_vec",
                data: Vec::new(),
            },
            cma: NamedVec {
                name: "cma_vec",
                data: Vec::new(),
            },
            cnpa: NamedVec {
                name: "cnpa_vec",
                data: Vec::new(),
            },
            cmadcmq: NamedVec {
                name: "cmadcmq_vec",
                data: Vec::new(),
            },
            cllp: NamedVec {
                name: "cllp_vec",
                data: Vec::new(),
            },
            cd_adim: NamedVec {
                name: "cd_adim_vec",
                data: Vec::new(),
            },
            cna_adim: NamedVec {
                name: "cna_adim_vec",
                data: Vec::new(),
            },
            cma_adim: NamedVec {
                name: "cma_adim_vec",
                data: Vec::new(),
            },
            cnpa_adim: NamedVec {
                name: "cnpa_adim_vec",
                data: Vec::new(),
            },
            cmadcmq_adim: NamedVec {
                name: "cmadcmq_adim_vec",
                data: Vec::new(),
            },
            cllp_adim: NamedVec {
                name: "cllp_adim_vec",
                data: Vec::new(),
            },
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
        self.vec_data.sp_vec.data = vec![0.0; self.iterations];
        self.vec_data.sp_vec.data = (0..self.iterations).map(|i| i as f64 * step).collect();
        let len = self.vec_data.sp_vec.data.len();
        self.vec_data.p_vec.data = vec![0.0; len];
        self.vec_data.p_vec.data[0] = self.roll_rate;
        self.vec_data.vel_vec.data = vec![0.0; len];
        self.vec_data.mach_vec.data = vec![0.0; len];
        self.vec_data.sg_vec.data = vec![0.0; len];
        self.vec_data.sd_vec.data = vec![0.0; len];
        self.vec_data.sg_lim_vec.data = vec![0.0; len];
        self.vec_data.dr_vec.data = vec![0.0; len];
        self.vec_data.alpha_vec.data = vec![0.0; len];
        self.vec_data.beta_vec.data = vec![0.0; len];
        self.vec_data.alpha_tot_vec.data = vec![0.0; len];
        self.vec_data.beta_r_vec.data = vec![0.0; len];
        self.vec_data.lambda_f_vec.data = vec![0.0; len];
        self.vec_data.lambda_s_vec.data = vec![0.0; len];
        self.vec_data.fast_freq_vec.data = vec![0.0; len];
        self.vec_data.slow_freq_vec.data = vec![0.0; len];
        self.vec_data.cd.data = vec![0.0; len];
        self.vec_data.cna.data = vec![0.0; len];
        self.vec_data.cma.data = vec![0.0; len];
        self.vec_data.cnpa.data = vec![0.0; len];
        self.vec_data.cmadcmq.data = vec![0.0; len];
        self.vec_data.cllp.data = vec![0.0; len];
        self.vec_data.cd_adim.data = vec![0.0; len];
        self.vec_data.cna_adim.data = vec![0.0; len];
        self.vec_data.cma_adim.data = vec![0.0; len];
        self.vec_data.cnpa_adim.data = vec![0.0; len];
        self.vec_data.cmadcmq_adim.data = vec![0.0; len];
        self.vec_data.cllp_adim.data = vec![0.0; len];
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

        self.vec_data.beta_vec.data[i] = xi.im();
        self.vec_data.alpha_vec.data[i] = xi.re();
        self.vec_data.alpha_tot_vec.data[i] = (xi.im().powf(2.0) + xi.re().powf(2.0)).sqrt();
        self.vec_data.beta_r_vec.data[i] = ((p * g) / (m + j * p * t)).re();
        self.vec_data.slow_freq_vec.data[i] =
            (phi_s * self.vec_data.vel_vec.data[i]) / (2.0 * PI * d);
        self.vec_data.fast_freq_vec.data[i] =
            (phi_f * self.vec_data.vel_vec.data[i]) / (2.0 * PI * d);
        self.vec_data.lambda_s_vec.data[i] = lam_s.re();
        self.vec_data.lambda_f_vec.data[i] = lam_f.re();
    }

    fn run(&mut self) {
        self.init_vectors();
        let j: Complex<f64> = Complex::new(0.0, 1.0);
        let kx_2: f64 = self.geometry.mass * self.geometry.diameter.powf(2.0) / self.geometry.in_x;
        let ky_2: f64 = self.geometry.mass * self.geometry.diameter.powf(2.0) / self.geometry.in_y;
        let inx_iny: f64 = self.geometry.in_x / self.geometry.in_y;
        let diam: f64 = self.geometry.diameter;

        self.update_aero(0, 0.0);

        let mut _p: f64 =
            inx_iny * ((self.vec_data.p_vec.data[0] * diam) / self.vec_data.vel_vec.data[0]);
        let mut _m: f64 = ky_2 * self.vec_data.cma.data[0];
        let mut _t: f64 = self.vec_data.cna.data[0];
        let mut _g: f64 = GRAV * diam * 0.0_f64.cos() / self.vec_data.vel_vec.data[0].powf(2.0);
        let mut _h: f64 = self.vec_data.cna.data[0]
            - self.vec_data.cd.data[0]
            - ky_2 * self.vec_data.cmadcmq.data[0];

        self.vec_data.sg_vec.data[0] = _p.powf(2.0) / (4.0 * _m);
        self.vec_data.sd_vec.data[0] = 2.0 * _t / _h;
        self.vec_data.sg_lim_vec.data[0] =
            1.0 / (self.vec_data.p_vec.data[0] * (2.0 - self.vec_data.sd_vec.data[0]));

        let eps: f64 = (1.0 - 1.0 / self.vec_data.sg_vec.data[0]).sqrt()
            * (self.delta_yaw.to_radians().sin())
            / (2.0 * (1.0 / inx_iny) - 1.0);

        let xi0 = eps.sin() * j.exp();
        let xi0_prime = j * ((self.roll_rate * diam) / self.init_vel) * xi0;

        let mut ttime = 0.0;
        for i in 0..self.vec_data.sp_vec.data.len() {
            let s = self.vec_data.sp_vec.data[i];
            if i == 0 {
                self.vec_data.vel_vec.data[i] = self.init_vel;
                self.update_aero(i, self.vec_data.alpha_tot_vec.data[0]);
            } else {
                self.vec_data.vel_vec.data[i] = self.vec_data.vel_vec.data[0]
                    * (-trapz(
                        &self.vec_data.cd_adim.data[0..i],
                        &self.vec_data.sp_vec.data[0..i],
                    ))
                    .exp();

                let delta_s = self.vec_data.sp_vec.data[i] - self.vec_data.sp_vec.data[i - 1];
                ttime = ttime + delta_s * diam / self.vec_data.vel_vec.data[i];

                let kp = -(kx_2 * self.vec_data.cllp_adim.data[i - 1]
                    + self.vec_data.cd_adim.data[i - 1]);

                self.vec_data.p_vec.data[i] = self.vec_data.vel_vec.data[i]
                    * self.vec_data.p_vec.data[0]
                    / self.vec_data.vel_vec.data[0]
                    * (-kp * self.vec_data.sp_vec.data[i]).exp();

                self.update_aero(i, self.vec_data.alpha_tot_vec.data[i - 1]);
            }

            _p = inx_iny * ((self.vec_data.p_vec.data[i] * diam) / self.vec_data.vel_vec.data[i]);
            _m = ky_2 * self.vec_data.cma_adim.data[i];
            _t = self.vec_data.cna_adim.data[i] + kx_2 * self.vec_data.cnpa_adim.data[i];
            _g = GRAV * diam * 0.0_f64.cos() / self.vec_data.vel_vec.data[i].powf(2.0);
            _h = self.vec_data.cna_adim.data[i]
                - self.vec_data.cd_adim.data[i]
                - ky_2 * self.vec_data.cmadcmq_adim.data[i];

            self.vec_data.sg_vec.data[i] = _p.powf(2.0) / (4.0 * _m);
            self.vec_data.sd_vec.data[i] = 2.0 * _t / _h;
            self.vec_data.sg_lim_vec.data[i] =
                1.0 / (self.vec_data.sd_vec.data[i] * (2.0 - self.vec_data.sd_vec.data[i]));

            self.vec_data.dr_vec.data[i] = (j / ky_2
                * (self.vec_data.p_vec.data[i] * _g / 2.0)
                * (self.vec_data.cna_adim.data[i] / self.vec_data.cma_adim.data[i])
                * s.powf(2.0)
                * (1.0
                    + 2.0 / 3.0 * (self.vec_data.cd_adim.data[i] * s)
                    + 1.0 / 3.0 * (self.vec_data.cd_adim.data[i] * s).powf(2.0)))
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
        self.vec_data.mach_vec.data[ind] = self.vec_data.vel_vec.data[ind] / self.sound;

        self.aero_data
            .update_coeffs(self.vec_data.mach_vec.data[ind]);

        self.vec_data.cd.data[ind] = self.aero_data.get_cd(alpha);
        self.vec_data.cna.data[ind] = self.aero_data.get_cna(alpha);
        self.vec_data.cma.data[ind] = self.aero_data.get_cma(alpha);
        self.vec_data.cnpa.data[ind] = self.aero_data.get_cnpa(alpha);
        self.vec_data.cmadcmq.data[ind] = self.aero_data.cmadcmq;
        self.vec_data.cllp.data[ind] = self.aero_data.cllp;

        self.vec_data.cd_adim.data[ind] = self.vec_data.cd.data[ind] * self.adim;
        self.vec_data.cna_adim.data[ind] = self.vec_data.cna.data[ind] * self.adim;
        self.vec_data.cma_adim.data[ind] = self.vec_data.cma.data[ind] * self.adim;
        self.vec_data.cnpa_adim.data[ind] = self.vec_data.cnpa.data[ind] * self.adim;
        self.vec_data.cmadcmq_adim.data[ind] = self.vec_data.cmadcmq.data[ind] * self.adim;
        self.vec_data.cllp_adim.data[ind] = self.vec_data.cllp.data[ind] * self.adim;
    }

    fn write_file(&self) {
        let file = Hdf5File::create("sim_res.hdf5").unwrap();

        write_datasets!(file, self.vec_data);

        // for (_, member) in self.vec_data.iter() {
        //     let item = member.downcast_ref::<NamedVec>().unwrap();
        //     file.new_dataset::<f64>()
        //         .shape(item.data.len())
        //         .create(item.name)
        //         .unwrap()
        //         .write(&item.data)
        //         .unwrap();
        // }
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
