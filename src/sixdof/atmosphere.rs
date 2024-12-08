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
    use crate::sixdof::atmosphere::Atmosphere;

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
