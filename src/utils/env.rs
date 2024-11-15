use pyo3::prelude::*;

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
