use pyo3::prelude::*;

#[pyclass]
pub struct Atmosphere {
    pub temperature: f64,
    pub sound_speed: f64,
    pub pressure: f64,
    pub density: f64,
    pub height: f64,
}

#[pymethods]
impl Atmosphere {
    #[new]
    pub fn new(altitude: f64) -> Self {
        let mut atmosphere = Atmosphere {
            temperature: 0.0,
            sound_speed: 0.0,
            pressure: 0.0,
            density: 0.0,
            height: altitude,
        };
        atmosphere.atmosisa(altitude);
        atmosphere
    }

    pub fn atmosisa(&mut self, altitude: f64) {
        self.height = altitude;

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
            if self.height < height_bound {
                level = i - 1;
                break;
            }
        }

        let delta = self.height - heights[level];

        let lapse =
            (temperatures[level + 1] - temperatures[level]) / (heights[level + 1] - heights[level]);

        self.temperature = temperatures[level] + delta * lapse;

        self.pressure = if lapse != 0.0 {
            // Non-zero lapse rate
            pressures[level]
                * (1.0 + lapse * delta / temperatures[level]).powf(-g * m / (r * lapse))
        } else {
            // Zero lapse rate (isothermal layer)
            pressures[level] * (-g * m * delta / (r * self.temperature)).exp()
        };

        self.density = self.pressure / (rs * self.temperature);

        self.sound_speed = 331.3 * (1.0 + (self.temperature - 273.15) / 273.15).sqrt();
    }
}
