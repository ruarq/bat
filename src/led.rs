type Rgb = rgb::Rgb<u8>;

pub struct LedStrip {
    pub data: Vec<Rgb>,
}

impl LedStrip {
    pub fn with_size(num_leds: usize) -> Self {
        Self {
            data: vec![Rgb { r: 0, g: 255, b: 0 }; num_leds],
        }
    }

    pub fn amp(&mut self, meter: f32) {
        assert!(meter >= 0.0 && meter <= 1.0);

        let lit_leds = (meter * self.data.len() as f32) as usize;

        for (i, rgb) in self.data.iter_mut().enumerate() {
            *rgb = if i < lit_leds {
                Rgb { r: 0, g: 255, b: 0 }
            } else {
                Rgb { r: 0, g: 0, b: 0 }
            };
        }
    }

    pub fn biamp(&mut self, meter: f32) {
        assert!(meter >= 0.0 && meter <= 1.0);

        // TODO: simplify this calculation
        let lit_leds = (meter * self.data.len() as f32) as usize / 2;
        let center_left = self.data.len() / 2 - 1;
        let center_right = self.data.len() / 2;
        let start = center_left - lit_leds;
        let end = center_right + lit_leds;

        for (i, rgb) in self.data.iter_mut().enumerate() {
            *rgb = if i > start && i < end {
                Rgb {
                    r: 255,
                    g: 255,
                    b: 0,
                }
            } else {
                Rgb { r: 0, g: 0, b: 0 }
            };
        }
    }

    pub fn spectrum(&mut self, spectrum: &[f32], frequencies: &[f32]) {
        assert_eq!(spectrum.len(), frequencies.len());

        let min_freq = frequencies[0];
        let max_freq = frequencies[frequencies.len() - 1];

        let num_leds = self.data.len();

        let map_freq_to_led_idx = |f: f32| {
            let range = max_freq.log10() - min_freq.log10();
            let x = (f.log10() - min_freq.log10()) / range;
            //eprintln!("{}, {}", min_freq, max_freq.log10());
            //eprintln!("{} - {} / {} = {}", f.log10(), min_freq.log10(), range, x);
            (x * (num_leds - 1) as f32) as usize
        };

        for (f, v) in frequencies.iter().zip(spectrum) {
            self.data[map_freq_to_led_idx(*f)] = Rgb {
                r: (0.0 * v) as u8,
                g: (0.0 * v) as u8,
                b: (100.0 * v) as u8,
            };
        }
    }
}
